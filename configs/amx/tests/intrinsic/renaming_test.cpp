// Tile renaming test in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/renaming_test.cpp, which explains it.
//
// Each test is one work region of Ops steps. The config prints the region's
// cycles at "--- END of AMX Region ---"; divide by Ops.

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

const int Ops = 960; // divisible by 3, the loop size below

alignas(64) TileConfig config;
alignas(64) uint8_t source[1024];

// Time test() as one work region. The region starts after all earlier AMX
// work and ends once the test's AMX work has finished. test() runs once
// untimed first, like on hardware, so the source is in L1.
void
region(const char *name, void (*test)())
{
    test();
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

    // One A tile: every load overwrites the tile the previous dot product
    // reads. Three A tiles: no load overwrites a tile still being read.
    region("one A tile", [] {
        for (int i = 0; i < Ops / 3; ++i) {
            amx_tile_loadd(3, source, 64);
            amx_tile_dpbf16ps(0, 3, 6);
            amx_tile_loadd(3, source, 64);
            amx_tile_dpbf16ps(1, 3, 6);
            amx_tile_loadd(3, source, 64);
            amx_tile_dpbf16ps(2, 3, 6);
        }
    });
    region("three A tiles", [] {
        for (int i = 0; i < Ops / 3; ++i) {
            amx_tile_loadd(3, source, 64);
            amx_tile_dpbf16ps(0, 3, 6);
            amx_tile_loadd(4, source, 64);
            amx_tile_dpbf16ps(1, 4, 6);
            amx_tile_loadd(5, source, 64);
            amx_tile_dpbf16ps(2, 5, 6);
        }
    });

    return 0;
}
