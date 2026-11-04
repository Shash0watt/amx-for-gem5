// TDPBF16PS latency and throughput in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/dot_product_test.cpp.
//
// Each test is one work region of Ops dot products. The config prints the
// region's cycles at "--- END of AMX Region ---"; divide by Ops.

#include <cstdint>
#include <cstdio>

#include <gem5/m5ops.h>

// The 64-byte memory layout read by LDTILECFG.
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

const int Ops = 960; // divisible by 2 and 6, the loop sizes below

alignas(64) TileConfig config;
alignas(64) uint16_t ones[512]; // one 16 x 64-byte tile of BF16 1.0

// Time test() as one work region. The region starts after all earlier AMX
// work and ends once the test's AMX work has finished.
void
region(const char *name, void (*test)())
{
    amx_wait();
    std::printf("%s\n", name);
    std::fflush(stdout); // print the name before the region marker
    m5_work_begin(0, 0);
    test();
    amx_wait(); // hardware: LFENCE
    m5_work_end(0, 0);
}

int
main()
{
    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.columnBytes[tile] = 64;
    }
    for (int i = 0; i < 512; ++i) {
        ones[i] = 0x3f80;
    }
    amx_tile_loadconfig(&config); // also sets every tile to zero

    // Latency: each dot product needs the previous one's result, through
    // C, A or B. The tiles are zero so the fed-back values can't overflow.
    region("TDPBF16PS: chain through C", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_dpbf16ps(0, 6, 7);
        }
    });
    region("TDPBF16PS: chain through A", [] {
        for (int i = 0; i < Ops / 2; ++i) {
            amx_tile_dpbf16ps(1, 0, 7);
            amx_tile_dpbf16ps(0, 1, 7);
        }
    });
    region("TDPBF16PS: chain through B", [] {
        for (int i = 0; i < Ops / 2; ++i) {
            amx_tile_dpbf16ps(1, 6, 0);
            amx_tile_dpbf16ps(0, 6, 1);
        }
    });

    // Throughput: six independent accumulators. Each is reused only every
    // six dot products, after its result is ready, so only the issue rate
    // limits them.
    amx_tile_loadd(6, ones, 64);
    amx_tile_loadd(7, ones, 64);
    region("TDPBF16PS: six accumulators", [] {
        for (int i = 0; i < Ops / 6; ++i) {
            amx_tile_dpbf16ps(0, 6, 7);
            amx_tile_dpbf16ps(1, 6, 7);
            amx_tile_dpbf16ps(2, 6, 7);
            amx_tile_dpbf16ps(3, 6, 7);
            amx_tile_dpbf16ps(4, 6, 7);
            amx_tile_dpbf16ps(5, 6, 7);
        }
    });

    return 0;
}
