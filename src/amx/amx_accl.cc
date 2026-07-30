#include "amx/amx_accl.hh"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "arch/generic/mmu.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "debug/AMX.hh"
#include "mem/port.hh"
#include "sim/faults.hh"

namespace gem5
{

// this creates the accelerator in the architectural init state.
AmxAccl::AmxAccl(const AmxAcclParams &params)
    : ClockedObject(params),
      memSidePort(name() + ".mem_side", *this),
      cpu(nullptr),
      currentCfg{},
      tilesConfigured(false),
      configLatency(params.config_latency),
      configCompletionEvent(
          [this] { finishConfigInstruction(configCompletionInstId); },
          name() + ".config_completion"),
      configCompletionInstId(0)
{
    clearTiles();

    DPRINTF(AMX, "Created the AMX SimObject\n");
}

// this records the cpu that owns the accelerator.
void
AmxAccl::setCPU(BaseCPU *_cpu)
{
    panic_if(!_cpu, "AMX accelerator requires a valid parent CPU");
    cpu = _cpu;
    DPRINTF(AMX, "The AMX accelerator is connected to parent CPU: %s\n",
            cpu->name());
}

// this exposes the accelerator memory port to python configuration code.
Port &
AmxAccl::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "mem_side") {
        return memSidePort;
    }

    return ClockedObject::getPort(if_name, idx);
}

// this reports that the accelerator is ready to receive work.
void
AmxAccl::startup()
{ DPRINTF(AMX, "AMX object started up\n"); }

// this creates the queued port used for accelerator reads.
AmxAccl::AmxRequestPort::AmxRequestPort(const std::string &name,
                                        AmxAccl &owner)
    : QueuedRequestPort(name, reqQueue, snoopRespQueue),
      owner(owner),
      reqQueue(owner, *this),
      snoopRespQueue(owner, *this) // unused but needed by a queued port
{}

// this forwards each cache response to the accelerator.
bool
AmxAccl::AmxRequestPort::recvTimingResp(PacketPtr pkt)
{
    owner.handleMemResponse(pkt);
    return true;
}

// this keeps byte-placement metadata with an asynchronous translation.
AmxAccl::AmxTranslation::AmxTranslation(AmxAccl &owner, uint64_t inst_id,
                                        const MemoryReadChunk &read_chunk)
    : owner(owner), instId(inst_id), readChunk(read_chunk)
{}

// this records that a translation is waiting on a page walk.
void
AmxAccl::AmxTranslation::markDelayed()
{
    // a full-system page walk calls this when translation is delayed
    // the translation counter keeps the instruction active until finish()
    DPRINTFS(AMX, &owner, "Translation delayed for instruction %llu\n",
             static_cast<unsigned long long>(instId));
}

// this returns a translation result to the accelerator and frees the callback.
void
AmxAccl::AmxTranslation::finish(const Fault &fault, const RequestPtr &req,
                                ThreadContext *, BaseMMU::Mode mode)
{
    panic_if(mode != BaseMMU::Read,
             "AMX load translation completed with a non-read mode");

    // the queue entry owns the thread context, so only placement data travels
    // through this one-shot callback
    owner.finishTranslation(instId, readChunk, fault, req);
    delete this;
}

// this adds a tile load to the accelerator queue.
void
AmxAccl::startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem,
                      uint64_t stride)
{
    panic_if(!tc, "AMX: Tile load requires a valid thread context");
    panic_if(dest_tile >= NUM_TILES,
             "AMX: Target tile %llu exceeds max tiles!",
             static_cast<unsigned long long>(dest_tile));
    DPRINTF(AMX, "Adding a LOAD for Tile %d to queue\n",
            static_cast<int>(dest_tile));

    const uint64_t id = nextInstId++;
    instructionQueue.emplace_back(id, AmxOpcode::AMX_LOAD,
                                  static_cast<int8_t>(dest_tile), -1, -1,
                                  src_mem, stride, tc);

    tryIssue();
}

// this adds a 64-byte tile configuration load to the accelerator queue.
void
AmxAccl::startAmxLoadConfig(ThreadContext *tc, uint64_t config_addr)
{
    panic_if(!tc, "AMX: Tile configuration requires a thread context");
    panic_if(config_addr > std::numeric_limits<uint64_t>::max() -
                               (TILE_CONFIG_BYTES - 1),
             "AMX: Tile configuration address wraps around");

    const uint64_t id = nextInstId++;
    instructionQueue.emplace_back(id, AmxOpcode::AMX_CONFIG, -1, -1, -1,
                                  config_addr, 0, tc);

    DPRINTF(AMX, "Adding a LOAD CFG for Tile %llu to queue\n",
            static_cast<unsigned long long>(id));
    tryIssue();
}

// this issues every queue entry that is currently free of hazards.
void
AmxAccl::tryIssue()
{
    // synchronous zero-sized loads can re-enter this function while it runs
    if (issuingInstructions) {
        return;
    }

    DPRINTF(AMX, "Queue: Finding next issuable instruction\n");
    issuingInstructions = true;
    AmxInst *ready_inst = nullptr;
    while ((ready_inst = findReadyInstruction()) != nullptr) {
        // no issue-bandwidth model exists, so release all hazard-free work
        executeInstruction(ready_inst);
    }
    issuingInstructions = false;

    if (!instructionQueue.empty()) {
        DPRINTF(AMX, "Queue: No issuable instruction found\n");
    }
}

// this finds the oldest instruction that can issue without breaking hazards.
AmxAccl::AmxInst *
AmxAccl::findReadyInstruction()
{
    const auto reads_tile = [](const AmxInst &inst, int tile) {
        if (inst.opcode == AmxOpcode::AMX_COMPUTE) {
            return inst.srcTile1 == tile || inst.srcTile2 == tile ||
                   inst.destTile == tile;
        }
        return inst.opcode == AmxOpcode::AMX_STORE && inst.srcTile1 == tile;
    };

    const auto writes_tile = [](const AmxInst &inst, int tile) {
        return (inst.opcode == AmxOpcode::AMX_LOAD ||
                inst.opcode == AmxOpcode::AMX_COMPUTE) &&
               inst.destTile == tile;
    };

    for (auto it = instructionQueue.begin(); it != instructionQueue.end();
         ++it) {
        AmxInst &inst = *it;

        if (inst.opcode == AmxOpcode::AMX_CONFIG) {
            // the config remains a barrier while it fetches and commits
            if (inst.state != AmxInst::State::PENDING) {
                return nullptr;
            }

            // reaching the front proves all older queue entries have drained
            if (it != instructionQueue.begin() || !allTilesIdle()) {
                return nullptr;
            }
            return &inst;
        }

        if (inst.state != AmxInst::State::PENDING) {
            continue;
        }

        bool has_hazard = false;
        for (int tile = 0; tile < NUM_TILES; ++tile) {
            if ((reads_tile(inst, tile) && tileScoreboard[tile].writeActive) ||
                (writes_tile(inst, tile) &&
                 (tileScoreboard[tile].writeActive ||
                  tileScoreboard[tile].readerCount > 0))) {
                has_hazard = true;
                break;
            }
        }

        // pending older work also reserves its dependencies in program order
        for (auto prior = instructionQueue.begin(); !has_hazard && prior != it;
             ++prior) {
            if (prior->state == AmxInst::State::COMPLETED) {
                continue;
            }
            for (int tile = 0; tile < NUM_TILES; ++tile) {
                if ((reads_tile(inst, tile) && writes_tile(*prior, tile)) ||
                    (writes_tile(inst, tile) && (reads_tile(*prior, tile) ||
                                                 writes_tile(*prior, tile)))) {
                    has_hazard = true;
                    break;
                }
            }
        }

        if (!has_hazard) {
            return &inst;
        }
    }

    return nullptr;
}

// this finds an in-flight instruction from its stable queue id.
AmxAccl::AmxInst *
AmxAccl::findInstruction(uint64_t inst_id)
{
    // look up callbacks by id instead of keeping a pointer into a changing
    // instruction queue
    for (auto &inst : instructionQueue) {
        if (inst.instId == inst_id) {
            return &inst;
        }
    }

    return nullptr;
}

// this checks that no tile reader or writer is still active.
bool
AmxAccl::allTilesIdle() const
{
    for (const auto &entry : tileScoreboard) {
        if (entry.writeActive || entry.readerCount != 0) {
            return false;
        }
    }
    return true;
}

// this sends a ready queue entry to its opcode-specific implementation.
void
AmxAccl::executeInstruction(AmxInst *ready_inst)
{
    panic_if(!ready_inst || ready_inst->state != AmxInst::State::PENDING,
             "AMX scheduler selected an invalid queue entry");
    panic_if(!cpu, "AMX instruction issued without an attached CPU");

    switch (ready_inst->opcode) {
        case AmxOpcode::AMX_CONFIG:
            executeConfigInstruction(ready_inst);
            break;
        case AmxOpcode::AMX_LOAD:
            executeLoadInstruction(ready_inst);
            break;
        case AmxOpcode::AMX_COMPUTE:
            panic_if(!tilesConfigured,
                     "AMX compute issued before tile configuration");
            panic_if(ready_inst->destTile < 0 ||
                         ready_inst->destTile >= NUM_TILES ||
                         ready_inst->srcTile1 < -1 ||
                         ready_inst->srcTile1 >= NUM_TILES ||
                         ready_inst->srcTile2 < -1 ||
                         ready_inst->srcTile2 >= NUM_TILES,
                     "AMX compute has an invalid tile operand");
            ready_inst->state = AmxInst::State::EXECUTING;
            tileScoreboard[ready_inst->destTile].writeActive = true;
            if (ready_inst->srcTile1 != -1) {
                tileScoreboard[ready_inst->srcTile1].readerCount++;
            }
            if (ready_inst->destTile != -1) {
                tileScoreboard[ready_inst->destTile].readerCount++;
            }
            if (ready_inst->srcTile2 != -1) {
                tileScoreboard[ready_inst->srcTile2].readerCount++;
            }
            // todo: implement compute execution and completion.
            break;
        case AmxOpcode::AMX_STORE:
            panic_if(!tilesConfigured,
                     "AMX store issued before tile configuration");
            panic_if(ready_inst->srcTile1 < -1 ||
                         ready_inst->srcTile1 >= NUM_TILES,
                     "AMX store has an invalid tile operand");
            ready_inst->state = AmxInst::State::EXECUTING;
            if (ready_inst->srcTile1 != -1) {
                tileScoreboard[ready_inst->srcTile1].readerCount++;
            }
            // todo: implement store execution and completion.
            break;
        default:
            panic("AMX scheduler selected an unknown opcode");
    }
}

// this reads each configured row into the selected tile.
void
AmxAccl::executeLoadInstruction(AmxInst *inst)
{
    panic_if(!tilesConfigured,
             "AMX tile load issued before tile configuration");
    panic_if(inst->destTile < 0 || inst->destTile >= NUM_TILES,
             "AMX tile load has invalid tile %d", inst->destTile);
    panic_if(!inst->tc, "AMX tile load has no thread context");

    const uint16_t num_rows = currentCfg.rows[inst->destTile];
    const uint16_t row_bytes = currentCfg.colsb[inst->destTile];
    panic_if(num_rows > MAX_ROWS || row_bytes > MAX_COLS_BYTES,
             "AMX tile %d has invalid configured dimensions", inst->destTile);

    DPRINTF(AMX,
            "Executing tile load %llu for Tile %d (%u rows, %u bytes/row)\n",
            static_cast<unsigned long long>(inst->instId), inst->destTile,
            num_rows, row_bytes);

    inst->state = AmxInst::State::EXECUTING;
    inst->outstandingTranslations = 0;
    inst->outstandingRequests = 0;
    inst->translationDispatchComplete = false;
    tileScoreboard[inst->destTile].writeActive = true;

    // clearing the physical tile makes every inactive byte read as zero
    tiles[inst->destTile] = {};

    const size_t cache_line_size = cpu->cacheLineSize();
    panic_if(cache_line_size == 0, "AMX requires a nonzero cache line size");

    for (uint8_t row = 0;
         row < num_rows && inst->failure == AmxInst::Failure::NONE; ++row) {
        if (row != 0) {
            panic_if(inst->stride >
                         (std::numeric_limits<uint64_t>::max() - inst->addr) /
                             row,
                     "AMX tile load row address wraps around");
        }
        const uint64_t row_addr = inst->addr + row * inst->stride;
        panic_if(row_bytes != 0 &&
                     row_addr > std::numeric_limits<uint64_t>::max() -
                                    (row_bytes - 1),
                 "AMX tile load row crosses the address limit");

        uint64_t current_addr = row_addr;
        size_t row_offset = 0;
        size_t remaining = row_bytes;
        while (remaining != 0 && inst->failure == AmxInst::Failure::NONE) {
            const uint64_t line_addr =
                current_addr - (current_addr % cache_line_size);
            const size_t source_offset = current_addr - line_addr;
            const size_t bytes =
                std::min(remaining, cache_line_size - source_offset);
            const MemoryReadChunk read_chunk = {
                MemoryTarget::TILE_ROW,
                static_cast<uint8_t>(inst->destTile),
                row,
                source_offset,
                row_offset,
                bytes,
            };

            // full-line reads preserve the existing cache timing behavior
            dispatchMemoryReadChunk(inst, line_addr, cache_line_size,
                                    read_chunk);
            row_offset += bytes;
            remaining -= bytes;
            if (remaining != 0) {
                current_addr += bytes;
            }
        }
    }

    const uint64_t inst_id = inst->instId;
    inst->translationDispatchComplete = true;
    maybeFinishMemoryInstruction(inst_id);
}

// this stages the complete configuration payload without changing tile state.
void
AmxAccl::executeConfigInstruction(AmxInst *inst)
{
    panic_if(instructionQueue.empty() || &instructionQueue.front() != inst,
             "AMX tile configuration issued away from the queue front");
    panic_if(!allTilesIdle(),
             "AMX tile configuration issued while a tile is active");
    panic_if(!inst->tc, "AMX tile configuration has no thread context");

    inst->state = AmxInst::State::EXECUTING;
    inst->issueTick = curTick();
    inst->outstandingTranslations = 0;
    inst->outstandingRequests = 0;
    inst->translationDispatchComplete = false;
    inst->completionScheduled = false;
    inst->configData.fill(0);

    const size_t cache_line_size = cpu->cacheLineSize();
    panic_if(cache_line_size == 0, "AMX requires a nonzero cache line size");

    uint64_t current_addr = inst->addr;
    size_t config_offset = 0;
    size_t remaining = TILE_CONFIG_BYTES;
    while (remaining != 0 && inst->failure == AmxInst::Failure::NONE) {
        const uint64_t line_addr =
            current_addr - (current_addr % cache_line_size);
        const size_t source_offset = current_addr - line_addr;
        const size_t bytes =
            std::min(remaining, cache_line_size - source_offset);
        const MemoryReadChunk read_chunk = {
            MemoryTarget::TILE_CONFIG,
            0,
            0,
            source_offset,
            config_offset,
            bytes,
        };

        dispatchMemoryReadChunk(inst, line_addr, cache_line_size, read_chunk);
        config_offset += bytes;
        remaining -= bytes;
        if (remaining != 0) {
            current_addr += bytes;
        }
    }

    const uint64_t inst_id = inst->instId;
    inst->translationDispatchComplete = true;
    maybeFinishMemoryInstruction(inst_id);
}

// this starts one timing translation for a cache-line-bounded memory read.
void
AmxAccl::dispatchMemoryReadChunk(AmxInst *inst, uint64_t virtual_addr,
                                 size_t request_size,
                                 const MemoryReadChunk &read_chunk)
{
    panic_if(!inst || !inst->tc, "AMX read has no thread context");
    panic_if(read_chunk.sourceOffset > request_size ||
                 read_chunk.bytesToCopy >
                     request_size - read_chunk.sourceOffset,
             "AMX memory read chunk exceeds its request");

    RequestPtr req = std::make_shared<Request>(
        virtual_addr, request_size, 0,
        inst->tc->getCpuPtr()->dataRequestorId(),
        inst->tc->pcState().instAddr(), inst->tc->contextId());

    // translation can finish immediately, so account for it before dispatch
    inst->outstandingTranslations++;
    auto *translation = new AmxTranslation(*this, inst->instId, read_chunk);
    inst->tc->getMMUPtr()->translateTiming(req, inst->tc, translation,
                                           BaseMMU::Read);
}

// this converts a successful translation into a timed cache read.
void
AmxAccl::finishTranslation(uint64_t inst_id,
                           const MemoryReadChunk &read_chunk,
                           const Fault &fault, const RequestPtr &req)
{
    AmxInst *inst = findInstruction(inst_id);
    panic_if(!inst, "AMX translation for unknown instruction %llu",
             static_cast<unsigned long long>(inst_id));
    panic_if(inst->state != AmxInst::State::EXECUTING,
             "AMX translation returned for an inactive instruction");
    panic_if(inst->outstandingTranslations == 0,
             "AMX instruction received too many translation callbacks");

    switch (read_chunk.target) {
        case MemoryTarget::TILE_ROW:
            panic_if(inst->opcode != AmxOpcode::AMX_LOAD ||
                         inst->destTile != read_chunk.tile,
                     "AMX tile read chunk has the wrong instruction owner");
            break;
        case MemoryTarget::TILE_CONFIG:
            panic_if(inst->opcode != AmxOpcode::AMX_CONFIG,
                     "AMX config read chunk has the wrong instruction owner");
            break;
        default:
            panic("AMX translation has an unknown memory target");
    }

    inst->outstandingTranslations--;

    if (fault != NoFault) {
        if (inst->failure == AmxInst::Failure::NONE) {
            inst->failure = AmxInst::Failure::TRANSLATION;
            inst->fault = fault;
        }
        maybeFinishMemoryInstruction(inst_id);
        return;
    }

    // after the first failure, later translations only need to drain
    if (inst->failure != AmxInst::Failure::NONE) {
        maybeFinishMemoryInstruction(inst_id);
        return;
    }

    panic_if(!req->hasPaddr(),
             "AMX translation for instruction %llu returned no paddr",
             static_cast<unsigned long long>(inst_id));
    req->setTranslateLatency();

    PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
    pkt->allocate();
    pkt->pushSenderState(new AmxSenderState(inst_id, read_chunk));

    // a response may return quickly, so count it before scheduling the packet
    inst->outstandingRequests++;
    memSidePort.schedTimingReq(pkt, curTick());

    DPRINTF(AMX,
            "Sent read for instruction %llu: vaddr 0x%lx -> paddr 0x%lx\n",
            static_cast<unsigned long long>(inst_id), req->getVaddr(),
            req->getPaddr());
    maybeFinishMemoryInstruction(inst_id);
}

// this completes a memory instruction after all of its callbacks have drained.
void
AmxAccl::maybeFinishMemoryInstruction(uint64_t inst_id)
{
    AmxInst *inst = findInstruction(inst_id);
    panic_if(!inst, "AMX completion check for unknown instruction %llu",
             static_cast<unsigned long long>(inst_id));

    if (!inst->translationDispatchComplete ||
        inst->outstandingTranslations != 0 || inst->outstandingRequests != 0) {
        return;
    }

    if (inst->opcode == AmxOpcode::AMX_LOAD) {
        finishLoadInstruction(inst);
        return;
    }

    panic_if(inst->opcode != AmxOpcode::AMX_CONFIG,
             "AMX memory completion reached an unsupported opcode");
    if (inst->failure != AmxInst::Failure::NONE) {
        finishConfigInstruction(inst_id);
        return;
    }
    if (inst->completionScheduled) {
        return;
    }

    panic_if(configCompletionEvent.scheduled(),
             "AMX scheduled two configuration completions");
    inst->completionScheduled = true;
    configCompletionInstId = inst_id;

    // memory latency overlaps the measured minimum instruction latency
    const Tick earliest = inst->issueTick + cyclesToTicks(configLatency);
    schedule(configCompletionEvent, std::max(clockEdge(), earliest));
}

// this releases a completed tile load and wakes the queue.
void
AmxAccl::finishLoadInstruction(AmxInst *inst)
{
    panic_if(!inst || inst->opcode != AmxOpcode::AMX_LOAD,
             "AMX load completion received the wrong instruction");
    panic_if(!inst->translationDispatchComplete ||
                 inst->outstandingTranslations != 0 ||
                 inst->outstandingRequests != 0,
             "AMX tile load completed with outstanding memory work");

    inst->state = AmxInst::State::COMPLETED;
    tileScoreboard[inst->destTile].writeActive = false;

    switch (inst->failure) {
        case AmxInst::Failure::TRANSLATION:
            panic("AMX: Tile load %llu failed address translation: %s. "
                  "Asynchronous fault delivery is not implemented.",
                  static_cast<unsigned long long>(inst->instId),
                  inst->fault->name());
        case AmxInst::Failure::MEMORY_ERROR:
            panic("AMX: Tile load %llu received an error response.",
                  static_cast<unsigned long long>(inst->instId));
        case AmxInst::Failure::MISSING_DATA:
            panic("AMX: Tile load %llu received a response without data.",
                  static_cast<unsigned long long>(inst->instId));
        case AmxInst::Failure::INVALID_CONFIG:
            panic("AMX: Tile load %llu has an internal configuration failure.",
                  static_cast<unsigned long long>(inst->instId));
        case AmxInst::Failure::NONE:
            break;
    }

    printInt32Tile(inst->destTile);
    const uint64_t completed_id = inst->instId;
    eraseInstruction(completed_id);
    tryIssue();
}

// this copies a cache response into its staged destination and drains it.
void
AmxAccl::handleMemResponse(PacketPtr pkt)
{
    panic_if(!pkt, "AMX received a null memory response");
    DPRINTF(AMX, "handleMemResponse called for packet at paddr 0x%lx\n",
            pkt->getAddr());

    auto *state = dynamic_cast<AmxSenderState *>(pkt->popSenderState());
    panic_if(!state,
             "amx response packet arrived missing its tracking senderstate!");

    AmxInst *inst = findInstruction(state->instId);
    panic_if(!inst, "AMX response for unknown instruction %llu",
             static_cast<unsigned long long>(state->instId));
    panic_if(inst->state != AmxInst::State::EXECUTING,
             "AMX response returned for an inactive instruction");
    panic_if(inst->outstandingRequests == 0,
             "AMX instruction %llu received too many responses",
             static_cast<unsigned long long>(inst->instId));

    if (pkt->isError()) {
        if (inst->failure == AmxInst::Failure::NONE) {
            inst->failure = AmxInst::Failure::MEMORY_ERROR;
        }
    }

    if (!pkt->isError() && !pkt->hasData()) {
        if (inst->failure == AmxInst::Failure::NONE) {
            inst->failure = AmxInst::Failure::MISSING_DATA;
        }
    }

    const MemoryReadChunk read_chunk = state->readChunk;
    panic_if(read_chunk.sourceOffset > pkt->getSize() ||
                 read_chunk.bytesToCopy >
                     pkt->getSize() - read_chunk.sourceOffset,
             "AMX response read chunk exceeds its packet");

    void *destination = nullptr;
    switch (read_chunk.target) {
        case MemoryTarget::TILE_ROW:
            panic_if(inst->opcode != AmxOpcode::AMX_LOAD,
                     "AMX tile response belongs to a non-load instruction");
            panic_if(read_chunk.tile >= NUM_TILES ||
                         read_chunk.row >= MAX_ROWS,
                     "AMX tile response has an invalid destination");
            panic_if(read_chunk.destOffset > MAX_COLS_BYTES ||
                         read_chunk.bytesToCopy >
                             MAX_COLS_BYTES - read_chunk.destOffset,
                     "AMX tile response exceeds its row");
            destination = &tiles[read_chunk.tile]
                               .data[read_chunk.row][read_chunk.destOffset];
            break;
        case MemoryTarget::TILE_CONFIG:
            panic_if(
                inst->opcode != AmxOpcode::AMX_CONFIG,
                "AMX config response belongs to a non-config instruction");
            panic_if(read_chunk.destOffset > TILE_CONFIG_BYTES ||
                         read_chunk.bytesToCopy >
                             TILE_CONFIG_BYTES - read_chunk.destOffset,
                     "AMX config response exceeds its staging buffer");
            destination = inst->configData.data() + read_chunk.destOffset;
            break;
        default:
            panic("AMX response has an unknown memory target");
    }

    if (inst->failure == AmxInst::Failure::NONE) {
        const uint8_t *source =
            pkt->getConstPtr<uint8_t>() + read_chunk.sourceOffset;
        std::memcpy(destination, source, read_chunk.bytesToCopy);
    }

    inst->outstandingRequests--;
    const uint64_t inst_id = state->instId;

    delete state;
    delete pkt;

    maybeFinishMemoryInstruction(inst_id);
}

// this validates and atomically commits a staged tile configuration.
void
AmxAccl::finishConfigInstruction(uint64_t inst_id)
{
    AmxInst *inst = findInstruction(inst_id);
    panic_if(!inst || inst->opcode != AmxOpcode::AMX_CONFIG,
             "AMX config completion received the wrong instruction");
    panic_if(!inst->translationDispatchComplete ||
                 inst->outstandingTranslations != 0 ||
                 inst->outstandingRequests != 0,
             "AMX tile configuration completed with outstanding memory work");
    panic_if(!allTilesIdle(),
             "AMX tile configuration completed while a tile is active");

    TileCfg candidate = {};
    if (inst->failure == AmxInst::Failure::NONE) {
        candidate = decodeTileConfig(inst->configData);
        if (!validateTileConfig(candidate, inst->failureReason)) {
            inst->failure = AmxInst::Failure::INVALID_CONFIG;
        }
    }

    // mark the entry terminal before reporting an unsupported guest fault
    inst->state = AmxInst::State::COMPLETED;
    switch (inst->failure) {
        case AmxInst::Failure::TRANSLATION:
            panic(
                "AMX: Tile configuration %llu failed address translation: %s. "
                "Asynchronous fault delivery is not implemented.",
                static_cast<unsigned long long>(inst->instId),
                inst->fault->name());
        case AmxInst::Failure::MEMORY_ERROR:
            panic("AMX: Tile configuration %llu received an error response.",
                  static_cast<unsigned long long>(inst->instId));
        case AmxInst::Failure::MISSING_DATA:
            panic("AMX: Tile configuration %llu received a response without "
                  "data.",
                  static_cast<unsigned long long>(inst->instId));
        case AmxInst::Failure::INVALID_CONFIG:
            panic(
                "AMX: Invalid tile configuration %llu: %s. Guest #GP delivery "
                "is not implemented.",
                static_cast<unsigned long long>(inst->instId),
                inst->failureReason.c_str());
        case AmxInst::Failure::NONE:
            break;
    }

    if (candidate.palette_id == 0) {
        currentCfg = TileCfg{};
        tilesConfigured = false;
    } else {
        currentCfg = candidate;
        tilesConfigured = true;
    }

    // both palette transitions architecturally clear all tile data
    clearTiles();
    DPRINTF(AMX, "Committed tile configuration %llu with palette %u\n",
            static_cast<unsigned long long>(inst_id),
            static_cast<unsigned>(candidate.palette_id));

    configCompletionInstId = 0;
    eraseInstruction(inst_id);
    tryIssue();
}

// this decodes the architectural little-endian 64-byte payload.
AmxAccl::TileCfg
AmxAccl::decodeTileConfig(
    const std::array<uint8_t, TILE_CONFIG_BYTES> &raw) const
{
    TileCfg config = {};
    config.palette_id = raw[0];
    config.start_row = raw[1];

    for (size_t i = 0; i < sizeof(config.reserved_0); ++i) {
        config.reserved_0[i] = raw[2 + i];
    }
    for (size_t tile = 0; tile < NUM_TILES; ++tile) {
        const size_t offset = 16 + tile * 2;
        config.colsb[tile] = static_cast<uint16_t>(raw[offset]) |
                             (static_cast<uint16_t>(raw[offset + 1]) << 8);
    }
    for (size_t i = 0; i < sizeof(config.reserved_1); ++i) {
        config.reserved_1[i] = raw[32 + i];
    }
    for (size_t tile = 0; tile < NUM_TILES; ++tile) {
        config.rows[tile] = raw[48 + tile];
    }
    for (size_t i = 0; i < sizeof(config.reserved_2); ++i) {
        config.reserved_2[i] = raw[56 + i];
    }

    return config;
}

// this checks the payload rules supported by the current single-palette model.
bool
AmxAccl::validateTileConfig(const TileCfg &config, std::string &reason) const
{
    const auto reject = [&reason](const std::string &message) {
        reason = message;
        return false;
    };

    if (config.palette_id != 0 && config.palette_id != 1) {
        return reject("unsupported palette " +
                      std::to_string(config.palette_id));
    }

    // restart state requires precise fault support, so only fresh loads work
    if (config.start_row != 0) {
        return reject("nonzero start_row is not supported");
    }

    for (const uint8_t byte : config.reserved_0) {
        if (byte != 0) {
            return reject("reserved bytes 2-15 must be zero");
        }
    }
    for (const uint8_t byte : config.reserved_1) {
        if (byte != 0) {
            return reject("reserved bytes 32-47 must be zero");
        }
    }
    for (const uint8_t byte : config.reserved_2) {
        if (byte != 0) {
            return reject("reserved bytes 56-63 must be zero");
        }
    }

    size_t total_bytes = 0;
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        const uint16_t rows = config.rows[tile];
        const uint16_t columns = config.colsb[tile];
        if (rows > MAX_ROWS) {
            return reject("tile " + std::to_string(tile) +
                          " has more than 16 rows");
        }
        if (columns > MAX_COLS_BYTES) {
            return reject("tile " + std::to_string(tile) +
                          " has more than 64 column bytes");
        }
        if ((rows == 0) != (columns == 0)) {
            return reject("tile " + std::to_string(tile) +
                          " has mismatched row and column dimensions");
        }
        if (config.palette_id == 0 && (rows != 0 || columns != 0)) {
            return reject("palette zero requires an empty tile layout");
        }
        total_bytes += rows * columns;
    }

    if (total_bytes > NUM_TILES * MAX_ROWS * MAX_COLS_BYTES) {
        return reject("tile layout exceeds the 8 kib register file");
    }
    return true;
}

// this resets every byte in the tile register file.
void
AmxAccl::clearTiles()
{
    for (auto &tile : tiles) {
        tile = {};
    }
}

// this removes one completed queue entry by its stable id.
void
AmxAccl::eraseInstruction(uint64_t inst_id)
{
    for (auto it = instructionQueue.begin(); it != instructionQueue.end();
         ++it) {
        if (it->instId == inst_id) {
            instructionQueue.erase(it);
            return;
        }
    }
    panic("AMX tried to erase unknown instruction %llu",
          static_cast<unsigned long long>(inst_id));
}

// this prints the active tile bytes to the amx debug trace.
void
AmxAccl::printInt8Tile(uint8_t tile_idx)
{
    panic_if(tile_idx >= NUM_TILES,
             "AMX printer: tile index %d out of bounds!", tile_idx);

    uint16_t active_rows = currentCfg.rows[tile_idx];
    uint16_t active_cols = currentCfg.colsb[tile_idx];

    std::stringstream ss;
    ss << "\n+================================================================"
          "========+\n";
    ss << "  AMX REGISTER STATE: [ TMM" << (int)tile_idx << " ] \n";
    ss << "  Layout Dimensions : " << active_rows << " Active Rows x "
       << active_cols << " Column Bytes\n";
    ss << "+=================================================================="
          "======+\n";

    for (uint8_t r = 0; r < active_rows; ++r) {
        // row labels
        ss << " Row [" << std::setw(2) << std::setfill('0') << std::dec
           << (int)r << "]: ";

        for (uint16_t c = 0; c < active_cols; ++c) {
            // read matrix register value
            int8_t val = tiles[tile_idx].data[r][c];
            ss << std::setw(4) << std::setfill(' ') << std::dec << (int)val
               << " ";
            if ((c + 1) % 4 == 0 && (c + 1) < active_cols) {
                ss << "| ";
            }
        }
        ss << "\n";
    }
    ss << "+=================================================================="
          "======+";

    // output dumped directly to the gem5 trace pipe
    DPRINTF(AMX, "%s\n", ss.str().c_str());
}

// this prints the active tile as packed 32-bit values to the debug trace.
void
AmxAccl::printInt32Tile(uint8_t tile_idx)
{
    panic_if(tile_idx >= NUM_TILES,
             "AMX printer: tile index %d out of bounds!", tile_idx);

    uint16_t active_rows = currentCfg.rows[tile_idx];
    uint16_t active_cols_bytes = currentCfg.colsb[tile_idx];
    uint16_t active_cols_32 = active_cols_bytes / 4;

    std::stringstream ss;
    ss << "\n+================================================================"
          "========+\n";
    ss << "  AMX REGISTER STATE: [ TMM" << (int)tile_idx << " ] \n";
    ss << "  Layout Dimensions : " << active_rows << " Active Rows x "
       << active_cols_32 << " Column Int32s\n";
    ss << "+=================================================================="
          "======+\n";

    for (uint8_t r = 0; r < active_rows; ++r) {
        // row labels
        ss << " Row [" << std::setw(2) << std::setfill('0') << std::dec
           << (int)r << "]: ";

        for (uint16_t c = 0; c < active_cols_32; ++c) {
            // copying avoids assuming that the byte-backed tile is aligned
            int32_t val = 0;
            std::memcpy(&val, &tiles[tile_idx].data[r][c * sizeof(val)],
                        sizeof(val));
            ss << std::setw(8) << std::setfill(' ') << std::dec << val << " ";
            if ((c + 1) % 4 == 0 && (c + 1) < active_cols_32) {
                ss << "| ";
            }
        }
        ss << "\n";
    }
    ss << "+=================================================================="
          "======+";

    // output dumped directly to the gem5 trace pipe
    DPRINTF(AMX, "%s\n", ss.str().c_str());
}

} // namespace gem5
