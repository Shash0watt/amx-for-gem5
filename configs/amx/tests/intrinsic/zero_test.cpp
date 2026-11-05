// TILEZERO latency and throughput in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/zero_test.cpp.
//
// Each test is one work region of Ops instructions. The config prints the
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

const int Ops = 960; // divisible by 8, the loop size below

alignas(64) TileConfig config;

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
    amx_tile_loadconfig(&config);

    // Latency: TILEZERO doesn't read its tile, so it can't form a chain.
    // Instead wait for each one to finish; the latency is "zero + wait"
    // minus "wait only".
    region("wait only (baseline)", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_wait(); // hardware: LFENCE
        }
    });
    region("TILEZERO: zero + wait", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_zero(0);
            amx_wait(); // hardware: LFENCE
        }
    });

    // Zero-to-use: "zero + TDPBF16PS + wait" minus "TDPBF16PS + wait",
    // zeroing the dot product's A and its C. Tiles 6 and 7 are zero from
    // the loadconfig.
    region("TDPBF16PS + LFENCE (baseline)", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_dpbf16ps(0, 6, 7);
            amx_wait(); // hardware: LFENCE
        }
    });
    region("zero A + TDPBF16PS + LFENCE", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_zero(1);
            amx_tile_dpbf16ps(0, 1, 7);
            amx_wait(); // hardware: LFENCE
        }
    });
    region("zero C + TDPBF16PS + LFENCE", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_zero(0);
            amx_tile_dpbf16ps(0, 6, 7);
            amx_wait(); // hardware: LFENCE
        }
    });

    // Throughput: eight different tiles, so no zero waits on another.
    region("TILEZERO: eight tiles", [] {
        for (int i = 0; i < Ops / 8; ++i) {
            amx_tile_zero(0);
            amx_tile_zero(1);
            amx_tile_zero(2);
            amx_tile_zero(3);
            amx_tile_zero(4);
            amx_tile_zero(5);
            amx_tile_zero(6);
            amx_tile_zero(7);
        }
    });

    return 0;
}
