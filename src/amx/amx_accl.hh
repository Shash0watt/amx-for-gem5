#ifndef __AMX_ACCL_HH__
#define __AMX_ACCL_HH__

#include "params/AmxAccl.hh"
#include "cpu/thread_context.hh"

// #include "sim/sim_object.hh" // swap out since we are using a clocked object
#include "sim/clocked_object.hh"

// #include "mem/port.hh"    // Required for RequestPort definition
#include "mem/packet.hh"  // Required for PacketPtr usage

namespace gem5
{

class BaseCPU;

class AmxAccl : public ClockedObject
{

  public:
    // used for async port req tracking with packets
    // we use this to identify exactly which tile and row the payload belongs to
    struct AmxSenderState : public Packet::SenderState
    {
        uint8_t destTile;
        uint8_t rowIdx;
        uint8_t offset;
        bool done;
        AmxSenderState(uint8_t dest, uint8_t row, uint8_t offset, bool done) :
            destTile(dest), rowIdx(row), offset(offset), done(done) {}
    };

    static constexpr int MAX_ROWS = 16;
    static constexpr int MAX_COLS_BYTES = 64;
    static constexpr int NUM_TILES = 8;

    struct TileCfg {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved_0[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    };

    struct TileReg {
        uint16_t rows;
        uint16_t colbytes;
        int8_t data[MAX_ROWS][MAX_COLS_BYTES];
    };

  private:
    // Pointer to the parent CPU. In core multiplexing, we access the cache pipeline
    // directly through the CPU's data port, removing the need for a separate RequestPort.
    BaseCPU *cpu;

    // internal registers for AMX. Moved from the deprecated AmxMemPort.
    TileCfg currentCfg;           // Global config state register
    TileReg tiles[NUM_TILES];     // Matrix register file (TMM0 - TMM7)

  public:
    AmxAccl(const AmxAcclParams &p);
    void startup() override;

    // Sets the parent CPU reference for accessing its memory ports.
    void setCPU(BaseCPU *_cpu);
    BaseCPU* getCPU() const { return cpu; }

    void startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem, uint64_t stride);

    // Handles memory responses routed from the CPU. Replaces AmxMemPort::recvTimingResp.
    void handleMemResponse(PacketPtr pkt);

    void printInt8Tile(uint8_t tile_idx);
};

} // namespace gem5

#endif // __AMX_ACCL_HH__
