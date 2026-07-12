#include "amx/amx_accl.hh"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "arch/generic/mmu.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "debug/AMX.hh"
#include "mem/port.hh"
// #include "params/AmxAccl.hh"

namespace gem5
{

AmxAccl::AmxAccl(const AmxAcclParams &params)
    : ClockedObject(params), cpu(nullptr), currentCfg{}
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

    DPRINTF(AMX, "created the amx object\n");
}

void
AmxAccl::setCPU(BaseCPU *_cpu)
{
    cpu = _cpu;
    DPRINTF(AMX, "parent cpu is set to %s\n", cpu->name());
}

void
AmxAccl::startup()
{ DPRINTF(AMX, "amx object startup completed\n"); }

void
AmxAccl::startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem,
                      std::size_t stride)
{
    // create a AmxInst object
    AmxInst load_inst = AmxInst(AMX_LOAD, dest_tile, -1, -1, src_mem, stride);

    // add it to the queue
    instructionQueue.push_back(load_inst);

    // try to issue the object
    tryIssue();
}

void
tryIssue()
{
    DPRINTF(AMX, "I really tried. I-I did!");

    // get the first instrution
    AmxInst &first_inst = instructionQueue.front();

    // execute it
    

    // pop it from queue
    instructionQueue.pop_front();

    // -- stuff to help me understand --
    // Look at the first instruction: AmxInst& first =
    // instructionQueue.front();

    // Remove the first instruction (once finished):
    // instructionQueue.pop_front();

    // Get the size: instructionQueue.size();

    // Iterate through the queue (for tryIssue):
    // for (auto& inst : instructionQueue) {
    //     // Check hazards for 'inst'
    // }
}

// void
// AmxAccl::startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t
// src_mem,
//                       std::size_t stride)
// {
//     // error checking
//     // make sure there are no out of bound accesses
//     panic_if(dest_tile >= NUM_TILES,
//              "AMX: Target tile %llu exceeds max tiles!", dest_tile);

//     uint16_t num_rows = currentCfg.rows[dest_tile];
//     uint16_t row_bytes = currentCfg.colsb[dest_tile];

//     DPRINTF(AMX,
//             "Executing amxload for Tile %llu (%d rows, %d bytes/row), Base "
//             "Src: 0x%llx, Stride: %lu\n",
//             dest_tile, num_rows, row_bytes, src_mem, stride);

//     // make sure we have the right ptr
//     if (!cpu) {
//         DPRINTF(AMX, "Warning the CPU is not attached / ptr is NULL\n");
//         return;
//     }

//     // get the cpu's dcache port
//     auto &dcache_port = dynamic_cast<RequestPort &>(cpu->getDataPort());
//     constexpr int CACHE_LINE_SIZE = 64; // Assuming 64-byte row width for
//     now

//     // reset the counter tracking for the tile's loads
//     tileOutstandingRequests[dest_tile] = 0;

//     // loop through each row in the tile
//     for (uint8_t r = 0; r < num_rows; ++r) {
//         // get the vaddr
//         uint64_t row_vaddr = src_mem + (r * stride);

//         size_t bytes_remaining = row_bytes;
//         uint16_t current_row_offset = 0;
//         uint64_t current_vaddr = row_vaddr;

//         while (bytes_remaining > 0) {
//             // make sure it's alligned to the cache line
//             uint64_t aligned_row_vaddr =
//                 current_vaddr & ~(CACHE_LINE_SIZE - 1);
//             uint8_t offset = current_vaddr & (CACHE_LINE_SIZE - 1);

//             // calculate bytes available in the current cache line block
//             size_t bytes_in_line = CACHE_LINE_SIZE - offset;
//             size_t chunk_size = std::min(bytes_remaining, bytes_in_line);

//             // make the request
//             RequestPtr req = std::make_shared<Request>(
//                 aligned_row_vaddr, CACHE_LINE_SIZE, 0,
//                 tc->getCpuPtr()->dataRequestorId(),
//                 tc->pcState().instAddr(), tc->contextId());

//             // get the virtual to physical address
//             Fault fault =
//                 tc->getMMUPtr()->translateFunctional(req, tc,
//                 BaseMMU::Read);
//             if (fault != NoFault) {
//                 DPRINTF(
//                     AMX,
//                     "Translation fault for Tile %llu, Row %d at Vaddr
//                     0x%lx\n", dest_tile, r, current_vaddr);
//                 // Stop dispatching if translation fails
//                 break;
//             }

//             // create the packet
//             PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
//             pkt->allocate(); // if the packet will carry data

//             // add the sender state information
//             pkt->pushSenderState(new AmxSenderState(
//                 dest_tile, r, offset, current_row_offset, chunk_size));

//             // Send the the packet to the CPU's memory port, make sure we do
//             // error handling
//             if (dcache_port.sendTimingReq(pkt)) {
//                 DPRINTF(AMX,
//                         "Dispatched row %d split read request. Paddr:
//                         0x%lx\n", r, req->getPaddr());
//                 // increment total requests expected back for this tile load
//                 tileOutstandingRequests[dest_tile]++;
//             } else {
//                 // NOTE: If the L1 cache queue is full, it rejects the
//                 packet. DPRINTF(AMX,
//                         "Port structural hazard! L1 Cache rejected row
//                         %d.\n", r);

//                 // clean up the rejected packet to avoid memory leaks
//                 delete pkt->popSenderState();
//                 delete pkt;
//                 // TODO: implement retry qeue... for now we just delete the
//                 // packet
//             }

//             bytes_remaining -= chunk_size;
//             current_row_offset += chunk_size;
//             current_vaddr += chunk_size;
//         }
//     }
// }

void
AmxAccl::handleMemResponse(PacketPtr pkt)
{
    DPRINTF(AMX, "amx: handleMemResponse called for packet at paddr 0x%lx\n",
            pkt->getAddr());

    // inline lambda function to delete packets without repeating code
    auto dropPacket = [](PacketPtr p) {
        if (p->senderState) {
            delete p->popSenderState();
        }
        delete p;
    };

    // check if the memory request failed
    if (pkt->isError()) {
        DPRINTF(AMX, "packet returned with an error status\n");
        dropPacket(pkt);
        return;
    }

    // check if the packet is empty
    if (!pkt->hasData()) {
        DPRINTF(AMX, "packet arrived safely but contains no data payload\n");
        dropPacket(pkt);
        return;
    }

    // get the sender state and the information from it
    auto *state = dynamic_cast<AmxSenderState *>(pkt->popSenderState());
    panic_if(
        !state,
        "amx response packet arrived missing its tracking senderstate token!");

    uint8_t tile = state->destTile;
    uint8_t row = state->rowIdx;
    uint8_t offset = state->cacheOffset;
    uint16_t row_offset = state->rowOffset;
    size_t copy_size = state->bytesToCopy;

    // error checking for the information
    panic_if(tile >= NUM_TILES, "AMX: Returned tile index %d out of bounds!",
             tile);
    panic_if(row >= MAX_ROWS, "AMX: Returned row index %d out of bounds!",
             row);
    panic_if((row_offset + copy_size) > MAX_COLS_BYTES,
             "AMX: Target row buffer boundary overflow!");

    const uint8_t *payload_start = pkt->getConstPtr<uint8_t>() + offset;

    // write data using the shifted pointer
    std::memcpy(&tiles[tile].data[row][row_offset], payload_start, copy_size);

    DPRINTF(AMX, "Loaded %zu bytes into Tile %d, Row %d (Offset: %d)\n",
            copy_size, tile, row, offset);

    // decrement outstanding packet tracking counter safely
    if (tileOutstandingRequests[tile] > 0) {
        tileOutstandingRequests[tile]--;
    }

    // check if done
    if (tileOutstandingRequests[tile] == 0) {
        printInt32Tile(tile);
    }

    // clean up allocated memory
    delete state;
    delete pkt;
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