// Queue a burst of L1D-hit loads from the CPU and from AMX so the
// l1d_amx_xbar trace shows how fast it serves them.
//
//   make all
//   ./build/X86/gem5.opt -rs --debug-flags=BaseXBar,CoherentXBar \
//       configs/amx/sapphire_rapids_config.py \
//       --binary configs/amx/binaries/xbar/xbar_load_test
//   grep l1d_amx_xbar m5out/simout.txt
//
// The AMX debug flag is only on inside the work region, so its lines mark
// where the region starts and ends in the trace.

#include <cstdint>

#include <gem5/m5ops.h>

// The 64-byte memory layout consumed by AMX_TILE_LOADCONFIG.
struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

// 16 independent loads, one per cache line. Loads into the same register
// don't depend on each other after renaming, so the O3 CPU can issue them
// all back to back.
static void
cpuLoadBurst(const uint8_t *lines)
{
    asm volatile(
        ".irp off,0,64,128,192,256,320,384,448,"
        "512,576,640,704,768,832,896,960\n"
        "mov \\off(%0), %%rax\n"
        ".endr\n"
        :
        : "r"(lines)
        : "rax", "memory");
}

int
main()
{
    // 16 rows x 64 bytes: a full tile, one cache line per row.
    alignas(64) static uint8_t cpuLines[16][64];
    alignas(64) static uint8_t tileRows[16][64];

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 64;
    config.rows[0] = 16;

    // Touch every line first so the timed loads all hit in the L1D.
    for (int row = 0; row < 16; ++row) {
        cpuLines[row][0] = row;
        tileRows[row][0] = row;
    }
    amx_tile_loadconfig(&config);
    amx_wait();

    m5_work_begin(0, 0);
    cpuLoadBurst(&cpuLines[0][0]);
    amx_tile_loadd(0, tileRows, sizeof(tileRows[0]));
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
