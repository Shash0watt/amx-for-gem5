#include "amx/amx_accl.hh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "arch/generic/mmu.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "debug/AMX.hh"
#include "mem/port.hh"
#include "sim/faults.hh"
// #include "params/AmxAccl.hh"

namespace gem5
{

AmxAccl::AmxAccl(const AmxAcclParams &params)
    : ClockedObject(params),
      memSidePort(name() + ".mem_side", *this), // initalize the memSidePort
      cpu(nullptr),
      currentCfg{}
{
    // initialize tiles array to zero
    for (int i = 0; i < NUM_TILES; i++) {
        tiles[i] = {};
    }

    // set up default configuration values
    currentCfg.palette_id = 1;
    currentCfg.start_row = 0;

    // set default row and column bounds for all tiles
    for (int i = 0; i < NUM_TILES; i++) {
        currentCfg.rows[i] = MAX_ROWS;
        currentCfg.colsb[i] = MAX_COLS_BYTES;
    }

    DPRINTF(AMX, "Created the AMX SimObject\n");
}

void
AmxAccl::setCPU(BaseCPU *_cpu)
{
    cpu = _cpu;
    DPRINTF(AMX, "The AMX accelerator is connected to parent CPU: %s\n",
            cpu->name());
}

Port &
AmxAccl::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "mem_side") {
        return memSidePort;
    }

    return ClockedObject::getPort(if_name, idx);
}

void
AmxAccl::startup()
{ DPRINTF(AMX, "AMX object started up\n"); }

// port logic
AmxAccl::AmxRequestPort::AmxRequestPort(const std::string &name,
                                        AmxAccl &owner)
    : QueuedRequestPort(name, reqQueue, snoopRespQueue),
      owner(owner),
      reqQueue(owner, *this),
      snoopRespQueue(owner, *this) // unused but needed by a queued port
{}

// callback for recived responses
bool
AmxAccl::AmxRequestPort::recvTimingResp(PacketPtr pkt)
{
    owner.handleMemResponse(pkt);
    return true;
}

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

    // generate a unique ID
    uint64_t id = nextInstId++;

    // create a AmxInst object
    AmxInst load_inst =
        AmxInst(id, AmxOpcode::AMX_LOAD, static_cast<int8_t>(dest_tile), -1,
                -1, src_mem, stride, tc);

    // add it to the queue
    instructionQueue.push_back(load_inst);

    // try to issue the object
    tryIssue();
}

void
AmxAccl::tryIssue()
{
    DPRINTF(AMX, "Queue: Finding next issuable instruction\n");

    // make sure there is an instruction to process
    if (instructionQueue.empty()) {
        DPRINTF(AMX, "Queue: Instruction queue is empty, nothing to issue\n");
        return;
    }

    AmxInst *ready_inst = findReadyInstruction();
    if (ready_inst != nullptr) {
        executeInstruction(ready_inst);
    } else {
        DPRINTF(AMX, "Queue: No issuable instruction found\n");
    }
}

AmxAccl::AmxInst *
AmxAccl::findReadyInstruction()
{
    AmxInst *ready_inst = nullptr;
    // all instructions in the queue
    for (auto it = instructionQueue.begin(); it != instructionQueue.end();
         ++it) {
        AmxInst &inst = *it;
        if (inst.state != AmxInst::State::PENDING) {
            continue;
        }
        bool has_hazard = false;
        // check with scoreboard
        // check that a tile is valid

        // read after write
        // I am doing a EXEC but my src1 and src2 or dest is being written to
        // I am doing a STR but my src is being written to.
        if (inst.opcode == AmxOpcode::AMX_COMPUTE) {
            if ((inst.srcTile1 != -1 &&
                 tileScoreboard[inst.srcTile1].writeActive) ||
                (inst.srcTile2 != -1 &&
                 tileScoreboard[inst.srcTile2].writeActive) ||
                (inst.destTile != -1 &&
                 tileScoreboard[inst.destTile].writeActive)) {
                has_hazard = true;
            }
        } else if (inst.opcode == AmxOpcode::AMX_STORE) {
            if (inst.srcTile1 != -1 &&
                tileScoreboard[inst.srcTile1].writeActive) {
                has_hazard = true;
            }
        }

        // write after write
        // I am doing a LOAD into a tile dest which is being written to
        // I am doing a EXEC and writing the result to a tile still being
        // written to does a store cause a WAW in gem5?
        if (inst.opcode == AmxOpcode::AMX_LOAD ||
            inst.opcode == AmxOpcode::AMX_COMPUTE) {
            if (inst.destTile != -1 &&
                tileScoreboard[inst.destTile].writeActive) {
                has_hazard = true;
            }
        }

        // write after read
        // I am doing a EXEC but a STR or other EXEC is reading it's dest, src1
        // or src2 I amd doing a LOAD but a STR's src or a EXEC's dest1, src1
        // or src2 is beign read from
        if (inst.opcode == AmxOpcode::AMX_LOAD ||
            inst.opcode == AmxOpcode::AMX_COMPUTE) {
            if (inst.destTile != -1 &&
                tileScoreboard[inst.destTile].readerCount > 0) {
                has_hazard = true;
            }
        }

        // check with previous pending instructions
        if (has_hazard == false) {
            // all instructions located before the current one (it/inst)
            for (auto prior_it = instructionQueue.begin(); prior_it != it;
                 ++prior_it) {
                AmxInst &prior = *prior_it;

                // skip checking against completed instructions
                if (prior.state == AmxInst::State::COMPLETED) {
                    continue;
                }

                // RAW
                if (inst.opcode == AmxOpcode::AMX_COMPUTE) {
                    if (prior.opcode == AmxOpcode::AMX_LOAD ||
                        prior.opcode == AmxOpcode::AMX_COMPUTE) {
                        if ((prior.destTile != -1 &&
                             prior.destTile == inst.srcTile1) ||
                            (prior.destTile != -1 &&
                             prior.destTile == inst.srcTile2) ||
                            (prior.destTile != -1 &&
                             prior.destTile == inst.destTile)) {
                            has_hazard = true;
                            break;
                        }
                    }
                } else if (inst.opcode == AmxOpcode::AMX_STORE) {
                    if (prior.opcode == AmxOpcode::AMX_LOAD ||
                        prior.opcode == AmxOpcode::AMX_COMPUTE) {
                        if (prior.destTile != -1 &&
                            prior.destTile == inst.srcTile1) {
                            has_hazard = true;
                            break;
                        }
                    }
                }

                // WAWA
                if (inst.opcode == AmxOpcode::AMX_LOAD ||
                    inst.opcode == AmxOpcode::AMX_COMPUTE) {
                    if (prior.opcode == AmxOpcode::AMX_LOAD ||
                        prior.opcode == AmxOpcode::AMX_COMPUTE) {
                        if (prior.destTile != -1 &&
                            prior.destTile == inst.destTile) {
                            has_hazard = true;
                            break;
                        }
                    }
                }

                // WAR
                if (inst.opcode == AmxOpcode::AMX_LOAD ||
                    inst.opcode == AmxOpcode::AMX_COMPUTE) {
                    if (prior.opcode == AmxOpcode::AMX_COMPUTE) {
                        if ((prior.srcTile1 != -1 &&
                             prior.srcTile1 == inst.destTile) ||
                            (prior.srcTile2 != -1 &&
                             prior.srcTile2 == inst.destTile) ||
                            (prior.destTile != -1 &&
                             prior.destTile == inst.destTile)) {
                            has_hazard = true;
                            break;
                        }
                    } else if (prior.opcode == AmxOpcode::AMX_STORE) {
                        if (prior.srcTile1 != -1 &&
                            prior.srcTile1 == inst.destTile) {
                            has_hazard = true;
                            break;
                        }
                    }
                }
            }
        }

        if (!has_hazard) {
            ready_inst = &inst;
            break; // we found a valid instruction!
        }
    }

    // check for hazards with already executing instructions
    // check for hazards with preceeding instructions in the queue

    return ready_inst;
}

void
AmxAccl::executeInstruction(AmxInst *ready_inst)
{
    // make sure that we can actually execute an instruction
    DPRINTF(AMX, "Queue: Found an instruction to exectue\n");
    // make sure that accesses are not out of bounds
    panic_if(ready_inst->destTile < 0 || ready_inst->destTile >= NUM_TILES,
             "AMX: Target tile %d exceeds max tiles!", ready_inst->destTile);

    // make sure that we have a CPU attached
    if (!cpu) {
        DPRINTF(AMX, "Queue: Warning CPU is not attached / ptr is NULL\n");
        return;
    }

    // get information about the tile from the config
    uint16_t num_rows = currentCfg.rows[ready_inst->destTile];
    uint16_t row_bytes = currentCfg.colsb[ready_inst->destTile];

    // execute it based on opcode
    switch (ready_inst->opcode) {
        case AmxOpcode::AMX_LOAD:
            DPRINTF(AMX,
                    "Queue: Executing amxload for Tile %d (%d rows, %d "
                    "bytes/row), "
                    "Base Src: 0x%lx, Stride: %lu\n",
                    ready_inst->destTile, num_rows, row_bytes,
                    ready_inst->addr, ready_inst->stride);

            // update the scoreboard
            tileScoreboard[ready_inst->destTile].writeActive = true;

            {

                constexpr int CACHE_LINE_SIZE = 64;

                // set the scoreboard information
                ready_inst->outstandingRequests = 0;
                ready_inst->state = AmxInst::State::EXECUTING;
                bool stop_dispatch = false;

                // loop through each row in the tile
                for (uint8_t r = 0; r < num_rows && !stop_dispatch; ++r) {
                    // get the vaddr
                    uint64_t row_vaddr =
                        ready_inst->addr + (r * ready_inst->stride);

                    size_t bytes_remaining = row_bytes;
                    uint16_t current_row_offset = 0;
                    uint64_t current_vaddr = row_vaddr;

                    while (bytes_remaining > 0) {
                        // make sure it's alligned to the cache line
                        uint64_t aligned_row_vaddr =
                            current_vaddr & ~(CACHE_LINE_SIZE - 1);
                        uint8_t offset = current_vaddr & (CACHE_LINE_SIZE - 1);

                        // calculate bytes available in the current cache
                        // line block
                        size_t bytes_in_line = CACHE_LINE_SIZE - offset;
                        size_t chunk_size =
                            std::min(bytes_remaining, bytes_in_line);

                        // make the request
                        RequestPtr req = std::make_shared<Request>(
                            aligned_row_vaddr, CACHE_LINE_SIZE, 0,
                            ready_inst->tc->getCpuPtr()->dataRequestorId(),
                            ready_inst->tc->pcState().instAddr(),
                            ready_inst->tc->contextId());

                        // get the virtual to physical address
                        // TODO: make this actually a timing request
                        Fault fault =
                            ready_inst->tc->getMMUPtr()->translateFunctional(
                                req, ready_inst->tc, BaseMMU::Read);
                        if (fault != NoFault) {
                            DPRINTF(
                                AMX,
                                "Queue: Translation fault for Tile %d, Row "
                                "%d at Vaddr 0x%lx\n",
                                ready_inst->destTile, r, current_vaddr);
                            ready_inst->failure =
                                AmxInst::Failure::TRANSLATION;
                            ready_inst->fault = fault;
                            // Stop dispatching if translation fails
                            stop_dispatch = true;
                            break;
                        }

                        // create the packet
                        PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
                        pkt->allocate(); // if the packet will carry data

                        // add the sender state information
                        pkt->pushSenderState(new AmxSenderState(
                            ready_inst->instId, ready_inst->destTile, r,
                            offset, current_row_offset, chunk_size));

                        // Send the packet to the CPU's memory port
                        ready_inst->outstandingRequests++;
                        memSidePort.schedTimingReq(pkt, curTick());

                        bytes_remaining -= chunk_size;
                        current_row_offset += chunk_size;
                        current_vaddr += chunk_size;
                    }
                }

                // cleanup if the instruction never issues because of queue
                // being full or transalation failing etc
                if (ready_inst->outstandingRequests == 0) {
                    finishLoadInstruction(ready_inst);
                }
            }
            break;

        case AmxOpcode::AMX_COMPUTE:
            DPRINTF(AMX, "Queue: Executing AMX compute operation\n");

            // update the scoreboarjd
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

            // TODO: we don't actually do matrix multiplication yet, to be
            // implemented
            // TODO: then after that we can accuratley model the delay for
            // the execution by stalling or sumn
            break;

        case AmxOpcode::AMX_STORE:
            DPRINTF(AMX, "Queue: Executing AMX store operation\n");

            // update the scoreboard
            if (ready_inst->srcTile1 != -1) {
                tileScoreboard[ready_inst->srcTile1].readerCount++;
            }

            // TODO: implement store logic
            break;

        default:
            panic("called unknown opcode");
    }
}

void
AmxAccl::finishLoadInstruction(AmxInst *inst)
{
    panic_if(inst->outstandingRequests != 0,
             "AMX: Completing instruction %llu with outstanding requests!",
             static_cast<unsigned long long>(inst->instId));

    inst->state = AmxInst::State::COMPLETED;
    tileScoreboard[inst->destTile].writeActive = false;

    switch (inst->failure) {
      case AmxInst::Failure::TRANSLATION:
        panic("AMX: Tile load instruction %llu failed address translation: "
              "%s. Asynchronous AMX fault delivery is not implemented.",
              static_cast<unsigned long long>(inst->instId),
              inst->fault->name());
      case AmxInst::Failure::MEMORY_ERROR:
        panic("AMX: Tile load instruction %llu received an error response.",
              static_cast<unsigned long long>(inst->instId));
      case AmxInst::Failure::MISSING_DATA:
        panic("AMX: Tile load instruction %llu received a response without "
              "data.", static_cast<unsigned long long>(inst->instId));
      case AmxInst::Failure::NONE:
        break;
    }

    printInt32Tile(inst->destTile);

    const uint64_t completed_id = inst->instId;
    for (auto it = instructionQueue.begin(); it != instructionQueue.end();
         ++it) {
        if (it->instId == completed_id) {
            instructionQueue.erase(it);
            break;
        }
    }

    tryIssue();
}

void
AmxAccl::handleMemResponse(PacketPtr pkt)
{
    DPRINTF(AMX, "handleMemResponse called for packet at paddr 0x%lx\n",
            pkt->getAddr());

    // get the sender state and the information from it
    auto *state = dynamic_cast<AmxSenderState *>(pkt->popSenderState());
    panic_if(!state,
             "amx response packet arrived missing its tracking senderstate!");

    // find the corresponding instruction in the instructionQueue
    AmxInst *inst = nullptr;
    for (auto &queued_inst : instructionQueue) {
        if (queued_inst.instId == state->instId) {
            inst = &queued_inst;
            break;
        }
    }
    panic_if(!inst, "AMX response for unknown instruction %llu",
             static_cast<unsigned long long>(state->instId));

    // check if the memory request failed
    if (pkt->isError()) {
        DPRINTF(AMX, "packet returned with an error status\n");
        if (inst->failure == AmxInst::Failure::NONE) {
            inst->failure = AmxInst::Failure::MEMORY_ERROR;
        }
    }

    // check if the packet is empty
    if (!pkt->isError() && !pkt->hasData()) {
        DPRINTF(AMX, "packet arrived safely but contains no data payload\n");
        if (inst->failure == AmxInst::Failure::NONE) {
            inst->failure = AmxInst::Failure::MISSING_DATA;
        }
    }

    uint8_t tile = state->destTile;
    uint8_t row = state->rowIdx;
    uint8_t offset = state->cacheOffset;
    uint16_t row_offset = state->rowOffset;
    size_t copy_size = state->bytesToCopy;

    // error checking for the information
    panic_if(tile >= NUM_TILES, "returned tile index %d out of bounds!", tile);
    panic_if(row >= MAX_ROWS, "returned row index %d out of bounds!", row);
    panic_if((row_offset + copy_size) > MAX_COLS_BYTES,
             "target row buffer boundary overflow!");

    if (!pkt->isError() && pkt->hasData()) {
        const uint8_t *payload_start =
            pkt->getConstPtr<uint8_t>() + offset;

        // write data using the shifted pointer
        std::memcpy(&tiles[tile].data[row][row_offset], payload_start,
                    copy_size);

        DPRINTF(AMX, "Loaded %u bytes into Tile %d, Row %d (Offset: %d)\n",
                (unsigned)copy_size, tile, row, offset);
    }

    panic_if(inst->outstandingRequests == 0,
             "AMX instruction %llu received too many responses",
             static_cast<unsigned long long>(inst->instId));
    inst->outstandingRequests--;
    const bool load_complete = inst->outstandingRequests == 0;

    // clean up allocated memory
    delete state;
    delete pkt;

    if (load_complete) {
        finishLoadInstruction(inst);
    }
}

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

        // cast the row data pointer to int32_t*
        const int32_t *row_data =
            reinterpret_cast<const int32_t *>(tiles[tile_idx].data[r]);

        for (uint16_t c = 0; c < active_cols_32; ++c) {
            // read matrix register value
            int32_t val = row_data[c];
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

// when starting AMX multiplication, when can we actually start?
// becuase the rows arrive out of order

// also what if data is not aligned to the cache line, what was the way that
// AMX does it

} // namespace gem5

// TODO: get out of order working again
// remember
// When  AMX_LOAD completes:
//     tileScoreboard[inst.destTile].writeActive = false;
// When  AMX_COMPUTE completes:
//     tileScoreboard[inst.destTile].writeActive = false;
//     tileScoreboard[inst.srcTile1].readerCount--;
//     tileScoreboard[inst.srcTile2].readerCount--;
//     tileScoreboard[inst.destTile].readerCount--;
// When  AMX_STORE completes:
//     tileScoreboard[inst.srcTile1].readerCount--;
