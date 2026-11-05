//   g++ -O2 -mamx-tile -mamx-bf16 -o renaming_test renaming_test.cpp
//   AMX_BENCH_CPU=<core> ./renaming_test

#include <cstdint>

#include "perf_helpers.h"

const int Ops = 960;

alignas(64) TileConfig config;
alignas(64) uint8_t source[1024];

int
main()
{
    setUpBenchmark();

    fillConfig(config, 16, 64);
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // Does hardware rename tiles? Each step loads an A tile and runs a dot
    // product on it into one of three accumulators (B is tile 6), so the dot
    // products don't wait for each other and could run at their issue
    // interval, 16 cycles per step.
    // "one A tile" always loads tile 3, so every load overwrites the tile
    // the previous dot product reads. Without renaming the load must wait
    // for that dot product to read tile 3, and the next dot product waits
    // for the load, so a step takes longer than 16.
    // "three A tiles" rotates through tiles 3, 4 and 5, so no load
    // overwrites a tile that is still being read.
    // Same time for both = the tile is renamed.
    region("one A tile", Ops, [] {
        for (int i = 0; i < Ops / 3; ++i) {
            _tile_loadd(3, source, 64);
            _tile_dpbf16ps(0, 3, 6);
            _tile_loadd(3, source, 64);
            _tile_dpbf16ps(1, 3, 6);
            _tile_loadd(3, source, 64);
            _tile_dpbf16ps(2, 3, 6);
        }
    });
    region("three A tiles", Ops, [] {
        for (int i = 0; i < Ops / 3; ++i) {
            _tile_loadd(3, source, 64);
            _tile_dpbf16ps(0, 3, 6);
            _tile_loadd(4, source, 64);
            _tile_dpbf16ps(1, 4, 6);
            _tile_loadd(5, source, 64);
            _tile_dpbf16ps(2, 5, 6);
        }
    });

    _tile_release();
    return 0;
}
