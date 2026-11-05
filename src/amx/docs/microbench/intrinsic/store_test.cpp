//   g++ -O2 -mamx-tile -mamx-bf16 -o store_test store_test.cpp
//   AMX_BENCH_CPU=<core> ./store_test

#include <cstdint>

#include "perf_helpers.h"

const int Ops = 960;

alignas(64) TileConfig config;
alignas(64) uint8_t destination[8 * 1024];

int
main()
{
    setUpBenchmark();

    fillConfig(config, 16, 64);
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // Latency: a store's address doesn't come from a tile, so stores can't
    // be chained. Instead an MFENCE after each one waits for its writes to
    // reach the cache (LFENCE doesn't wait for stores); latency = "store +
    // wait" minus "wait only". The same 1 KiB every time, so it stays in L1.
    region("wait only (baseline)", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _mm_mfence();
        }
    });
    region("TILESTORED: store + wait, L1", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_stored(0, destination, 64);
            _mm_mfence();
        }
    });

    // Throughput: stores of eight different tiles to eight 1 KiB buffers
    // back to back, so none waits for another and cycles per op = the issue
    // interval. The MFENCE at the end makes the timing include the writes.
    region("TILESTORED: eight tiles, L1", Ops, [] {
        for (int i = 0; i < Ops / 8; ++i) {
            _tile_stored(0, destination + 0 * 1024, 64);
            _tile_stored(1, destination + 1 * 1024, 64);
            _tile_stored(2, destination + 2 * 1024, 64);
            _tile_stored(3, destination + 3 * 1024, 64);
            _tile_stored(4, destination + 4 * 1024, 64);
            _tile_stored(5, destination + 5 * 1024, 64);
            _tile_stored(6, destination + 6 * 1024, 64);
            _tile_stored(7, destination + 7 * 1024, 64);
        }
        _mm_mfence();
    });

    _tile_release();
    return 0;
}
