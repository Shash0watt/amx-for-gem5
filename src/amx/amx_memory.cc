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

// -------------------------------------------------------------------------
// Memory-instruction dispatch
// -------------------------------------------------------------------------

void
AmxAccl::beginMemoryInstruction(AmxInst *instruction)
{
    instruction->state = AmxInst::State::Executing;
    instruction->outstandingTranslations = 0;
    instruction->outstandingRequests = 0;
    instruction->translationDispatchComplete = false;
    instruction->completionScheduled = false;
}

void
AmxAccl::finishMemoryDispatch(AmxInst *instruction)
{
    const uint64_t instruction_id = instruction->id;
    instruction->translationDispatchComplete = true;
    maybeFinishMemoryInstruction(instruction_id);
}

void
AmxAccl::dispatchMemoryRead(AmxInst *instruction, uint64_t virtual_address,
                            size_t bytes, MemoryTarget target, uint8_t tile,
                            uint8_t row)
{
    panic_if(bytes != 0 &&
                 virtual_address >
                     std::numeric_limits<uint64_t>::max() - (bytes - 1),
             "AMX memory read crosses the address limit");

    const size_t cache_line_size = cpu->cacheLineSize();
    panic_if(cache_line_size == 0, "AMX requires a nonzero cache line size");

    uint64_t current_address = virtual_address;
    size_t destination_offset = 0;
    size_t remaining = bytes;

    while (remaining != 0 && instruction->failure == AmxInst::Failure::None) {
        const uint64_t line_address =
            current_address - (current_address % cache_line_size);
        const size_t source_offset = current_address - line_address;
        const size_t chunk_bytes =
            std::min(remaining, cache_line_size - source_offset);

        dispatchMemoryReadChunk(instruction, line_address, cache_line_size,
                                {target, tile, row, source_offset,
                                 destination_offset, chunk_bytes});

        destination_offset += chunk_bytes;
        remaining -= chunk_bytes;
        if (remaining != 0) {
            current_address += chunk_bytes;
        }
    }
}

void
AmxAccl::dispatchMemoryReadChunk(AmxInst *instruction,
                                 uint64_t virtual_address, size_t request_size,
                                 const MemoryReadChunk &read_chunk)
{
    panic_if(!instruction || !instruction->threadContext,
             "AMX read has no thread context");
    panic_if(read_chunk.sourceOffset > request_size ||
                 read_chunk.bytesToCopy >
                     request_size - read_chunk.sourceOffset,
             "AMX memory read chunk exceeds its request");

    ThreadContext *tc = instruction->threadContext;
    RequestPtr request = std::make_shared<Request>(
        virtual_address, request_size, 0, tc->getCpuPtr()->dataRequestorId(),
        tc->pcState().instAddr(), tc->contextId());

    // Translation can complete synchronously, so count it first.
    instruction->outstandingTranslations++;
    auto *translation = new AmxTranslation(*this, instruction->id, read_chunk);
    tc->getMMUPtr()->translateTiming(request, tc, translation, BaseMMU::Read);
}

// -------------------------------------------------------------------------
// Address translation
// -------------------------------------------------------------------------

AmxAccl::AmxTranslation::AmxTranslation(AmxAccl &owner,
                                        uint64_t instruction_id,
                                        const MemoryReadChunk &read_chunk)
    : owner(owner), instructionId(instruction_id), readChunk(read_chunk)
{}

void
AmxAccl::AmxTranslation::markDelayed()
{
    DPRINTFS(AMX, &owner, "Translation delayed for instruction %llu\n",
             static_cast<unsigned long long>(instructionId));
}

void
AmxAccl::AmxTranslation::finish(const Fault &fault, const RequestPtr &request,
                                ThreadContext *, BaseMMU::Mode mode)
{
    panic_if(mode != BaseMMU::Read,
             "AMX load translation completed with a non-read mode");

    owner.finishTranslation(instructionId, readChunk, fault, request);
    delete this;
}

void
AmxAccl::finishTranslation(uint64_t instruction_id,
                           const MemoryReadChunk &read_chunk,
                           const Fault &fault, const RequestPtr &request)
{
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction, "AMX translation for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));
    panic_if(instruction->state != AmxInst::State::Executing,
             "AMX translation returned for an inactive instruction");
    panic_if(instruction->outstandingTranslations == 0,
             "AMX instruction received too many translation callbacks");

    validateMemoryReadOwner(*instruction, read_chunk);
    instruction->outstandingTranslations--;

    if (fault != NoFault) {
        if (instruction->failure == AmxInst::Failure::None) {
            instruction->failure = AmxInst::Failure::Translation;
            instruction->fault = fault;
        }
        maybeFinishMemoryInstruction(instruction_id);
        return;
    }

    // Once one operation fails, remaining translations only need to drain.
    if (instruction->failure != AmxInst::Failure::None) {
        maybeFinishMemoryInstruction(instruction_id);
        return;
    }

    panic_if(!request->hasPaddr(),
             "AMX translation for instruction %llu returned no paddr",
             static_cast<unsigned long long>(instruction_id));
    request->setTranslateLatency();

    PacketPtr packet = new Packet(request, MemCmd::ReadReq);
    packet->allocate();
    packet->pushSenderState(new AmxSenderState(instruction_id, read_chunk));

    // A response may return immediately, so count it before scheduling it.
    instruction->outstandingRequests++;
    memSidePort.schedTimingReq(packet, curTick());

    DPRINTF(AMX,
            "Sent read for instruction %llu: vaddr 0x%lx -> paddr 0x%lx\n",
            static_cast<unsigned long long>(instruction_id),
            request->getVaddr(), request->getPaddr());
    maybeFinishMemoryInstruction(instruction_id);
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

    if (packet->isError() && instruction->failure == AmxInst::Failure::None) {
        instruction->failure = AmxInst::Failure::MemoryError;
    } else if (!packet->hasData() &&
               instruction->failure == AmxInst::Failure::None) {
        instruction->failure = AmxInst::Failure::MissingData;
    }

    const MemoryReadChunk &read_chunk = state->readChunk;
    panic_if(read_chunk.sourceOffset > packet->getSize() ||
                 read_chunk.bytesToCopy >
                     packet->getSize() - read_chunk.sourceOffset,
             "AMX response read chunk exceeds its packet");

    void *destination = memoryDestination(*instruction, read_chunk);
    if (instruction->failure == AmxInst::Failure::None) {
        const uint8_t *source =
            packet->getConstPtr<uint8_t>() + read_chunk.sourceOffset;
        std::memcpy(destination, source, read_chunk.bytesToCopy);
    }

    instruction->outstandingRequests--;
    const uint64_t instruction_id = state->instructionId;

    delete state;
    delete packet;

    maybeFinishMemoryInstruction(instruction_id);
}

void
AmxAccl::validateMemoryReadOwner(const AmxInst &instruction,
                                 const MemoryReadChunk &read_chunk) const
{
    switch (read_chunk.target) {
        case MemoryTarget::TileRow:
            panic_if(instruction.opcode != AmxOpcode::Load ||
                         instruction.destination != read_chunk.tile,
                     "AMX tile read has the wrong instruction owner");
            return;
        case MemoryTarget::TileConfig:
            panic_if(instruction.opcode != AmxOpcode::Config,
                     "AMX config read has the wrong instruction owner");
            return;
    }

    panic("AMX memory read has an unknown target");
}

void *
AmxAccl::memoryDestination(AmxInst &instruction,
                           const MemoryReadChunk &read_chunk)
{
    validateMemoryReadOwner(instruction, read_chunk);

    switch (read_chunk.target) {
        case MemoryTarget::TileRow:
            panic_if(read_chunk.tile >= NUM_TILES ||
                         read_chunk.row >= MAX_ROWS,
                     "AMX tile response has an invalid destination");
            panic_if(read_chunk.destinationOffset > MAX_COLS_BYTES ||
                         read_chunk.bytesToCopy >
                             MAX_COLS_BYTES - read_chunk.destinationOffset,
                     "AMX tile response exceeds its row");
            return &tiles[read_chunk.tile]
                        .data[read_chunk.row][read_chunk.destinationOffset];

        case MemoryTarget::TileConfig:
            panic_if(read_chunk.destinationOffset > TILE_CONFIG_BYTES ||
                         read_chunk.bytesToCopy >
                             TILE_CONFIG_BYTES - read_chunk.destinationOffset,
                     "AMX config response exceeds its staging buffer");
            return instruction.configData.data() +
                   read_chunk.destinationOffset;
    }

    panic("AMX memory response has an unknown target");
}

// -------------------------------------------------------------------------
// Memory-instruction completion
// -------------------------------------------------------------------------

void
AmxAccl::maybeFinishMemoryInstruction(uint64_t instruction_id)
{
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction, "AMX completion check for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));

    if (!instruction->translationDispatchComplete ||
        instruction->outstandingTranslations != 0 ||
        instruction->outstandingRequests != 0) {
        return;
    }

    if (instruction->opcode == AmxOpcode::Load) {
        finishLoadInstruction(instruction);
        return;
    }

    panic_if(instruction->opcode != AmxOpcode::Config,
             "AMX memory completion reached an unsupported opcode");

    if (instruction->failure != AmxInst::Failure::None) {
        finishConfigInstruction(instruction_id);
        return;
    }
    if (instruction->completionScheduled) {
        return;
    }

    panic_if(configCompletionEvent.scheduled(),
             "AMX scheduled two configuration completions");
    instruction->completionScheduled = true;
    configCompletionInstructionId = instruction_id;

    // Memory latency overlaps the minimum configuration latency.
    const Tick earliest =
        instruction->issueTick + cyclesToTicks(configLatency);
    schedule(configCompletionEvent, std::max(clockEdge(), earliest));
}

} // namespace gem5
