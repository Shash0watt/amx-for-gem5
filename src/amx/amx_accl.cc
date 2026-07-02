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
    // error checking
    // make sure there are no out of bound accesses
    panic_if(dest_tile >= NUM_TILES,
             "AMX: Target tile %llu exceeds max tiles!", dest_tile);

    uint16_t num_rows = currentCfg.rows[dest_tile];
    uint16_t row_bytes = currentCfg.colsb[dest_tile];

    DPRINTF(AMX,
            "Executing amxload for Tile %llu (%d rows, %d bytes/row), Base "
            "Src: 0x%llx, Stride: %lu\n",
            dest_tile, num_rows, row_bytes, src_mem, stride);

    // make sure we have the right ptr
    if (!cpu) {
        DPRINTF(AMX, "Warning the CPU is not attached / ptr is NULL\n");
        return;
    }

    // get the cpu's dcache port
    auto &dcache_port = dynamic_cast<RequestPort &>(cpu->getDataPort());
    constexpr int CACHE_LINE_SIZE = 64; // Assuming 64-byte row width for now

    // loop through each row in the tile
    for (uint8_t r = 0; r < num_rows; ++r) {
        // get the vaddr
        uint64_t row_vaddr = src_mem + (r * stride);

        // make sure it's alligned to the cache line
        uint64_t aligned_row_vaddr = row_vaddr & ~(CACHE_LINE_SIZE - 1);

        // make the request
        RequestPtr req = std::make_shared<Request>(
            aligned_row_vaddr, CACHE_LINE_SIZE, 0,
            tc->getCpuPtr()->dataRequestorId(), tc->pcState().instAddr(),
            tc->contextId());

        // get the virtual to physical address
        Fault fault =
            tc->getMMUPtr()->translateFunctional(req, tc, BaseMMU::Read);
        if (fault != NoFault) {
            DPRINTF(AMX,
                    "Translation fault for Tile %llu, Row %d at Vaddr 0x%lx\n",
                    dest_tile, r, row_vaddr);
            // Stop dispatching if translation fails
            break;
        }

        // create the packet
        PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
        pkt->allocate(); // if the packet will carry data

        // add the sender state information
        pkt->pushSenderState(new AmxSenderState(dest_tile, r));

        // Send the the packet to the CPU's memory port, make sure we do error
        // handling
        if (dcache_port.sendTimingReq(pkt)) {
            DPRINTF(AMX, "Dispatched row %d read request. Paddr: 0x%lx\n", r,
                    req->getPaddr());
        } else {
            // NOTE: If the L1 cache queue is full, it rejects the packet.
            DPRINTF(AMX, "Port structural hazard! L1 Cache rejected row %d.\n",
                    r);

            // clean up the rejected packet to avoid memory leaks
            delete pkt->popSenderState();
            delete pkt;
            // TODO: implement retry qeue... for now we just delete the packet
        }
    }
}


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

    // extract and verify the tracking state from the packet
    auto *state = dynamic_cast<AmxSenderState *>(pkt->popSenderState());
    panic_if(
        !state,
        "amx response packet arrived missing its tracking senderstate token!");

    // convert raw packet bytes into a readable int8 string for debugging
    auto *data_ptr = reinterpret_cast<int8_t *>(pkt->getPtr<uint8_t>());
    std::string int8_output;
    for (int i = 0; i < pkt->getSize(); i++) {
        int8_output += std::to_string(data_ptr[i]) + " ";
    }

    DPRINTF(AMX, "data loaded into matrix (as int8): [ %s]\n",
            int8_output.c_str());

    // clean up allocated memory
    delete state;
    delete pkt;
}


} // namespace gem5