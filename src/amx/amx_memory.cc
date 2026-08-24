#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "cpu/thread_context.hh"
#include "debug/AMX.hh"

namespace gem5
{

// at a high level a instruction follows this path
// 1. mark the instrcution as exectuing (reads, writes & pending completion)
// 2. split each access at cache-line boundaries
// (becaause you can only make one request per line)
// 3. address translation on the vaddr
// 4. send the request through the gem5 system
// ---
// 5. gem5 callback -> copy returned read data or retire a completed write
// 6. mark the instruction as completed

// -------------------------------------------------------------------------
// Memory-instruction dispatch
// -------------------------------------------------------------------------

void
AmxAccl::beginMemoryInstruction(AmxInst *instruction)
{

    // Mark as executing because we can only tell completion by the callback
    // and counters
    instruction->state = AmxInst::State::Executing;

    // Counters, make sure that we cannot finish until all translation
    // callbacks and all resulting memory responses have drained.
    instruction->outstandingTranslations = 0;
    instruction->outstandingRequests = 0;

    // A translation callback is allowed to run synchronously while the caller
    // is still creating the rest of the reads.
    instruction->translationDispatchComplete = false;
    instruction->completionScheduled = false;
    instruction->memoryComplete = false;
}

void
AmxAccl::finishMemoryDispatch(AmxInst *instruction)
{
    // Save the id to use for the completion check.
    const uint64_t instruction_id = instruction->id;

    // Mark that no more 'work' will be created
    instruction->translationDispatchComplete = true;
    completeInstructionIfReady(instruction_id);
}

// this is the actual memory request
void
AmxAccl::dispatchMemoryRead(AmxInst *instruction, uint64_t virtual_address,
                            size_t bytes, uint8_t *host_dest)
{
    panic_if(bytes != 0 &&
                 virtual_address >
                     std::numeric_limits<uint64_t>::max() - (bytes - 1),
             "AMX memory read crosses the address limit");

    // AMX rows and the 64-byte tile configuration can start in the middle of
    // a cache line or cross line boundaries, Issue one request for every
    // cache line touched so the accesses behave like ordinary CPU cache-line
    // traffic in gem5's memory hierarchy.
    const size_t cache_line_size = cpu->cacheLineSize();
    panic_if(cache_line_size == 0, "AMX requires a nonzero cache line size");

    // `current_address` is for walking the guest-visible byte range.
    // `destination_offset` is for walking the output buffer (a tile row or
    // config staging buffer), independent of cache-line boundaries.
    uint64_t current_address = virtual_address;
    size_t destination_offset = 0;
    size_t remaining = bytes;

    while (remaining != 0 && instruction->failure == AmxInst::Failure::None) {
        // Get the starting address of the containing cache line.
        // The memoryrequest fetches the whole line so we need to keep track of
        // packetOffset/bytes to know which portion of the returned cache
        // line belongs to our AMX load
        const uint64_t line_address =
            current_address - (current_address % cache_line_size);
        const size_t source_offset = current_address - line_address;
        const size_t chunk_bytes =
            std::min(remaining, cache_line_size - source_offset);

        // This is the instruction along with all the metadata with this
        // cache-line operation needed by other stages
        dispatchMemoryChunk(instruction, line_address, cache_line_size,
                            {MemoryAccess::Read, host_dest + destination_offset,
                             source_offset, chunk_bytes});

        destination_offset += chunk_bytes;
        remaining -= chunk_bytes;
        if (remaining != 0) {
            current_address += chunk_bytes;
        }
    }
}

void
AmxAccl::dispatchMemoryWrite(AmxInst *instruction, uint64_t virtual_address,
                             size_t bytes, const uint8_t *host_src)
{
    panic_if(bytes != 0 &&
                 virtual_address >
                     std::numeric_limits<uint64_t>::max() - (bytes - 1),
             "AMX memory write crosses the address limit");

    const size_t cache_line_size = cpu->cacheLineSize();
    panic_if(cache_line_size == 0, "AMX requires a nonzero cache line size");

    uint64_t current_address = virtual_address;
    size_t source_offset = 0;
    size_t remaining = bytes;

    while (remaining != 0 && instruction->failure == AmxInst::Failure::None) {
        const size_t line_offset = current_address % cache_line_size;
        const size_t chunk_bytes =
            std::min(remaining, cache_line_size - line_offset);

        // Writes cover only the bytes architecturally selected by colsb.
        // Requesting an entire line here would overwrite adjacent guest data.
        dispatchMemoryChunk(
            instruction, current_address, chunk_bytes,
            {MemoryAccess::Write,
             const_cast<uint8_t *>(host_src + source_offset), 0, chunk_bytes});

        source_offset += chunk_bytes;
        remaining -= chunk_bytes;
        if (remaining != 0) {
            current_address += chunk_bytes;
        }
    }
}

void
AmxAccl::dispatchMemoryChunk(AmxInst *instruction,
                             uint64_t virtual_address, size_t request_size,
                             const MemoryChunk &memory_chunk)
{
    // These are internal invariants.  Violating them indicates an AMX model
    // bug rather than a guest-program error, so gem5 should stop immediately.
    panic_if(!instruction || !instruction->threadContext,
             "AMX memory access has no thread context");
    panic_if(memory_chunk.packetOffset > request_size ||
                 memory_chunk.bytes >
                     request_size - memory_chunk.packetOffset,
             "AMX memory chunk exceeds its request");
    panic_if(memory_chunk.access == MemoryAccess::Write &&
                 (memory_chunk.packetOffset != 0 ||
                  memory_chunk.bytes != request_size),
             "AMX write chunk must exactly cover its request");

    ThreadContext *tc = instruction->threadContext;

    // We need to first translate out address to a physical one ot make a cach
    // request
    RequestPtr request = std::make_shared<Request>(
        virtual_address, request_size, 0, tc->getCpuPtr()->dataRequestorId(),
        tc->pcState().instAddr(), tc->contextId());

    // since we are waiting for a callback we need to mark that we have a
    // pending translation
    instruction->outstandingTranslations++;

    // we make a translation request that has our custom callback
    auto *translation =
        new AmxTranslation(*this, instruction->id, memory_chunk);

    // to perform a timing adadress translation, get the MMU from the thread
    // context and use it's translate timing function so that we can get
    // accurate page table walks and etc
    const BaseMMU::Mode mode = memory_chunk.access == MemoryAccess::Read ?
        BaseMMU::Read : BaseMMU::Write;
    tc->getMMUPtr()->translateTiming(request, tc, translation, mode);
}

// -------------------------------------------------------------------------
// Address translation
// -------------------------------------------------------------------------

// Construct only the state that gem5's MMU will return to when translation
// finishes; creating the callback does not itself start translation.
AmxAccl::AmxTranslation::AmxTranslation(AmxAccl &owner,
                                        uint64_t instruction_id,
                                        const MemoryChunk &memory_chunk)
    : owner(owner), instructionId(instruction_id), memoryChunk(memory_chunk)
{}

void
AmxAccl::AmxTranslation::markDelayed()
{
    // gem5 calls markDelayed() when translation cannot complete immediately
    // (for example, while modeling a TLB miss/page-table walk).  For our
    // implementation no change is needed here because the
    // outstanding-translation count already keeps the parent instruction alive
    // until finish() arrives.
    DPRINTFS(AMX, &owner, "Translation delayed for instruction %llu\n",
             static_cast<unsigned long long>(instructionId));
}

void
AmxAccl::AmxTranslation::finish(const Fault &fault, const RequestPtr &request,
                                ThreadContext *, BaseMMU::Mode mode)
{
    // This is out custom callback that we sent with the translation request
    const BaseMMU::Mode expected_mode =
        memoryChunk.access == MemoryAccess::Read ?
            BaseMMU::Read : BaseMMU::Write;
    panic_if(mode != expected_mode,
             "AMX memory translation completed with the wrong mode");

    owner.finishTranslation(instructionId, memoryChunk, fault, request);

    delete this;
}

void
AmxAccl::finishTranslation(uint64_t instruction_id,
                           const MemoryChunk &memory_chunk,
                           const Fault &fault, const RequestPtr &request)
{
    // We can look the instruction up again instead of carrying a pointer
    // across the MMU operation.
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction, "AMX translation for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));
    panic_if(instruction->state != AmxInst::State::Executing,
             "AMX translation returned for an inactive instruction");
    panic_if(instruction->outstandingTranslations == 0,
             "AMX instruction received too many translation callbacks");

    // Our translation is done, so we can mark it as complete
    instruction->outstandingTranslations--;

    if (fault != NoFault) {
        if (instruction->failure == AmxInst::Failure::None)
        {
            instruction->failure = AmxInst::Failure::Translation;
            instruction->fault = fault;
        }
        // this could be our last translation, so we should try and see if we
        // can complete
        completeInstructionIfReady(instruction_id);
        return;
    }

    // A recorded failure is reported after the remaining memory work drains.
    if (instruction->failure != AmxInst::Failure::None) {
        completeInstructionIfReady(instruction_id);
        return;
    }

    panic_if(!request->hasPaddr(),
             "AMX translation for instruction %llu returned no paddr",
             static_cast<unsigned long long>(instruction_id));

    // Tell gem5 that the translation phase has ended
    request->setTranslateLatency();

    //  gem5's memory ports exchange Packets, not Requests directly.  The
    //  Request describes the access and its translated addresses; the Packet
    //  adds the command, response data buffer, and routing/ownership state
    //  used while the access travels through the timing memory system.
    const MemCmd command = memory_chunk.access == MemoryAccess::Read ?
        MemCmd::ReadReq : MemCmd::WriteReq;
    PacketPtr packet = new Packet(request, command);

    // Reads need a return buffer. Writes also use packet-owned storage so the
    // tile's bytes remain valid independently of later simulator activity.
    packet->allocate();
    if (memory_chunk.access == MemoryAccess::Write) {
        std::memcpy(packet->getPtr<uint8_t>() + memory_chunk.packetOffset,
                    memory_chunk.hostBuf, memory_chunk.bytes);
    }

    // SenderState is gem5's  way to carry component-private metadata, This is
    // what lets us find what instruction the returned data corresponds to.
    packet->pushSenderState(
        new AmxSenderState(instruction_id, memory_chunk));

    // A gem5 timing response may return immediately, so count it before the
    // queued port is allowed to send the packet.
    instruction->outstandingRequests++;

    // schedTimingReq() is the QueuedRequestPort form of a timing send.  It
    // submits the packet at the current simulated tick and handles request-
    // port backpressure/retries if the downstream cache is busy.
    memSidePort.schedTimingReq(packet, curTick());

    DPRINTF(AMX,
            "Sent %s for instruction %llu: vaddr 0x%lx -> paddr 0x%lx\n",
            memory_chunk.access == MemoryAccess::Read ? "read" : "write",
            static_cast<unsigned long long>(instruction_id),
            request->getVaddr(), request->getPaddr());
    completeInstructionIfReady(instruction_id);
}

// -------------------------------------------------------------------------
// Cache response handling
// -------------------------------------------------------------------------

void
AmxAccl::handleMemoryResponse(PacketPtr packet)
{

    panic_if(!packet, "AMX received a null memory response");
    DPRINTF(AMX, "Received memory response for paddr 0x%lx\n",
            packet->getAddr());

    // Recover and remove the private metadata attached before the request was
    // sent. popSenderState() is needed with gem5 because packets can carry
    // a stack of sender states added by multiple components along their path.
    auto *state = dynamic_cast<AmxSenderState *>(packet->popSenderState());
    panic_if(!state, "AMX response is missing its sender state");

    AmxInst *instruction = findInstruction(state->instructionId);
    panic_if(!instruction, "AMX response for unknown instruction %llu",
             static_cast<unsigned long long>(state->instructionId));
    panic_if(instruction->state != AmxInst::State::Executing,
             "AMX response returned for an inactive instruction");
    panic_if(instruction->outstandingRequests == 0,
             "AMX instruction %llu received too many responses",
             static_cast<unsigned long long>(instruction->id));

    const MemoryChunk &memory_chunk = state->memoryChunk;
    panic_if(!packet->isError() &&
                 memory_chunk.access == MemoryAccess::Read &&
                 !packet->isRead(),
             "AMX read received a non-read response");
    panic_if(!packet->isError() &&
                 memory_chunk.access == MemoryAccess::Write &&
                 !packet->isWrite(),
             "AMX write received a non-write response");

    // A timing response may signal an error or, unexpectedly for a read,
    // arrive without a data payload. Write responses need not carry data.
    // Record only the first failure so later responses merely drain the
    // instruction's outstanding work.
    if (packet->isError() && instruction->failure == AmxInst::Failure::None) {
        instruction->failure = AmxInst::Failure::MemoryError;
    } else if (memory_chunk.access == MemoryAccess::Read &&
               !packet->hasData() &&
               instruction->failure == AmxInst::Failure::None) {
        instruction->failure = AmxInst::Failure::MissingData;
    }

    panic_if(memory_chunk.packetOffset > packet->getSize() ||
                 memory_chunk.bytes >
                     packet->getSize() - memory_chunk.packetOffset,
             "AMX response chunk exceeds its packet");

    if (memory_chunk.access == MemoryAccess::Read &&
        instruction->failure == AmxInst::Failure::None) {
        // The packet contains a whole cache line.  Skip any bytes before the
        // original virtual address and copy only this logical AMX chunk.
        const uint8_t *source =
            packet->getConstPtr<uint8_t>() + memory_chunk.packetOffset;
        std::memcpy(memory_chunk.hostBuf, source, memory_chunk.bytes);
    }

    // Save what we need for the completion check before releasing the
    // per-packet objects
    instruction->outstandingRequests--;
    const uint64_t instruction_id = state->instructionId;

    delete state;
    delete packet;

    // possible that we have a finished instruction at the end of handling the
    // memory response
    completeInstructionIfReady(instruction_id);
}

} // namespace gem5
