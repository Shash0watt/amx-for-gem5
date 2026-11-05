//   g++ -O2 -mamx-tile -mamx-bf16 -o config_test config_test.cpp
//   AMX_BENCH_CPU=<core> ./config_test

#include "perf_helpers.h"

const int Ops = 960;

alignas(64) TileConfig fullTiles;
alignas(64) TileConfig halfTiles;

int
main()
{
    setUpBenchmark();

    fillConfig(fullTiles, 16, 64);
    fillConfig(halfTiles, 8, 32);
    asm volatile("" ::: "memory");

    // Latency: a config doesn't read a tile, so configs can't be chained.
    // Instead an LFENCE after each one waits for it; latency = "config +
    // wait" minus "wait only".
    region("wait only (baseline)", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _mm_lfence();
        }
    });
    region("LDTILECFG: config + wait", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_loadconfig(&fullTiles);
            _mm_lfence();
        }
    });

    // Throughput: configs back to back, cycles per op = the issue interval.
    region("LDTILECFG: same config", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_loadconfig(&fullTiles);
        }
    });
    region("LDTILECFG: alternating configs", Ops, [] {
        for (int i = 0; i < Ops / 2; ++i) {
            _tile_loadconfig(&fullTiles);
            _tile_loadconfig(&halfTiles);
        }
    });

    _tile_release();
    return 0;
}
