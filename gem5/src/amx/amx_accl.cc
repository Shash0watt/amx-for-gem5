#include "amx/amx_accl.hh"
#include <iostream>

// Needed for debug flags
#include "base/trace.hh" 
#include "debug/AMX.hh"
#include "cpu/base.hh"       
#include "arch/generic/mmu.hh"

namespace gem5
{

AmxAccl::AmxAccl(const AmxAcclParams &params) :
    ClockedObject(params),
    cpu(nullptr) // Pointer to the parent CPU, initialized as null and set via setCPU() by the CPU itself
{
    std::memset(&currentCfg, 0, sizeof(TileCfg));
    std::memset(tiles, 0, sizeof(tiles));

    // struct TileCfg {
    //         uint8_t palette_id;
    //         uint8_t start_row;
    //         uint8_t reserved_0[14];
    //         uint16_t colsb[16];
    //         uint8_t rows[16];
    //     };

    // for debugging and so that we don't need the tile config instruction right now
    currentCfg.palette_id = 1; // this means that the AMX accelerator will be on..
    // TODO implement palette_id check
    currentCfg.start_row = 0; // TODO implement start_row behaviour

    for (int i = 0; i < NUM_TILES; i++) {
        currentCfg.rows[i] = MAX_ROWS;         
        currentCfg.colsb[i] = MAX_COLS_BYTES;   
    }

    DPRINTF(AMX, "Created the AMX object\n");
}

void
AmxAccl::setCPU(BaseCPU *_cpu)
{
    cpu = _cpu;
    DPRINTF(AMX, "Parent CPU is set to %s\n", cpu->name());
}

void
AmxAccl::handleMemResponse(PacketPtr pkt)
{
    DPRINTF(AMX, "handleMemResponse called for packet at paddr 0x%lx\n", pkt->getAddr());

    if (pkt->isError()) {
        DPRINTF(AMX, "Packet returned with an ERROR status, Destination Address was unmapped or faulty.\n");
        delete pkt;
        return; 
    }

    if (!pkt->hasData()) {
        DPRINTF(AMX, "Packet arrived safely, but contains NO data payload to print.\n");
        delete pkt;
        return;
    }

    // Interpret the packet data buffer as signed 8-bit integers
    int8_t *data_ptr = reinterpret_cast<int8_t*>(pkt->getPtr<uint8_t>());

    std::string int8_output = "";
    for (int i = 0; i < pkt->getSize(); i++) {
        char buf[8];
        snprintf(buf, sizeof(buf), "%d ", data_ptr[i]);
        int8_output += buf;
    }
    
    DPRINTF(AMX, "Data loaded into matrix (as int8): [ %s]\n", int8_output.c_str());

    delete pkt;
}

void
AmxAccl::startup()
{
    DPRINTF(AMX, "AMX Object startup completed\n");
}

void
AmxAccl::startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem, size_t stride)
{
    DPRINTF(AMX, "Received amxLoadd.. Dest: %llu, Src: %llu, Stride: %lu\n", 
            dest_tile, src_mem, stride);

    // Internal core multiplexing debug log
    if (cpu) {
        RequestPort &dcache_port = dynamic_cast<RequestPort &>(cpu->getDataPort());
        DPRINTF(AMX, "Accessing CPU's dcache port: %s\n", dcache_port.name());
    } else {
        DPRINTF(AMX, "Warning, CPU pointer is null in startAmxLoad!\n");
    }

    // 1. Align the virtual address to a standard 64-byte cache line
    int cache_line_size = 64;     
    uint64_t aligned_src_mem = src_mem & ~(cache_line_size - 1);

    // 2. Construct a baseline master Request object
    // We grab the RequestorID, instruction pointer, and context ID directly from the CPU thread.
    RequestPtr req = std::make_shared<Request>(
        aligned_src_mem,
        cache_line_size,
        0,                                 // Flags
        tc->getCpuPtr()->dataRequestorId(),// ID from the attached core
        tc->pcState().instAddr(),          // The current instruction PC
        tc->contextId()                    // The thread ID
    );

    // TODO: perform a timing page-walk/address translation 
    Fault fault = tc->getMMUPtr()->translateFunctional(req, tc, BaseMMU::Read);
    if (fault != NoFault) {
        DPRINTF(AMX, "Translation failed for virtual address 0x%lx\n", src_mem);
        return; 
    }

    // do a strided load
    // uint8_t num_rows = currentCfg.num_rows[dest_tile];
    // uint8_t bytes_per_row = currentCfg.bytes_per_row[dest_tile];

    // for (int i = 0; i < num_rows; i++){
    //     // calculate the vaddr for each row
    //     // make the packet
    //     // send the request
    //     // what is sender state??
    // }

    // make a Read Packet
    PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
    pkt->allocate(); // used to initialize a data buffer within the packet

    // For the barebones phase, print debug messages and try to send the request via CPU's dcache port.
    // Instead of our deprecated amxMemPort, we now push requests directly onto the parent CPU's dcache_port.
    if (cpu) {
        RequestPort &dcache_port = dynamic_cast<RequestPort &>(cpu->getDataPort());
        if (dcache_port.sendTimingReq(pkt)) {
            DPRINTF(AMX, "Timing read request successfully sent via CPU dcache port for physical address 0x%lx\n", 
                    req->getPaddr());
        } else {
            DPRINTF(AMX, "CPU dcache port rejected the timing request (busy). Retries are not handled yet\n");
            delete pkt;
        }
    } else {
        DPRINTF(AMX, "CPU pointer not set, dropping packet\n");
        delete pkt;
    }
}

} // namespace gem5