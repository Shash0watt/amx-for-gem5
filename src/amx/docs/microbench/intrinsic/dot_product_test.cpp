//   g++ -O2 -mamx-tile -mamx-bf16 -o dot_product_test dot_product_test.cpp
//   AMX_BENCH_CPU=<core> ./dot_product_test

#include <cstdint>

#include "perf_helpers.h"

const int Ops = 960;

alignas(64) TileConfig config;
alignas(64) uint16_t ones[512];

int
main()
{
    setUpBenchmark();

    fillConfig(config, 16, 64);
    for (int i = 0; i < 512; ++i) {
        ones[i] = 0x3f80;
    }
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // Latency through each operand: every dot product reads the result of
    // the one before as its C, A or B, so cycles per op = how long that
    // operand waits for the previous result.
    region("TDPBF16PS: chain through C", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_dpbf16ps(0, 6, 7);
        }
    });
    region("TDPBF16PS: chain through A", Ops, [] {
        for (int i = 0; i < Ops / 2; ++i) {
            _tile_dpbf16ps(1, 0, 7);
            _tile_dpbf16ps(0, 1, 7);
        }
    });
    region("TDPBF16PS: chain through B", Ops, [] {
        for (int i = 0; i < Ops / 2; ++i) {
            _tile_dpbf16ps(1, 6, 0);
            _tile_dpbf16ps(0, 6, 1);
        }
    });

    // Throughput: six dot products into six different C tiles, so none
    // waits for another's result and cycles per op = the issue interval.
    _tile_loadd(6, ones, 64);
    _tile_loadd(7, ones, 64);
    region("TDPBF16PS: six accumulators", Ops, [] {
        for (int i = 0; i < Ops / 6; ++i) {
            _tile_dpbf16ps(0, 6, 7);
            _tile_dpbf16ps(1, 6, 7);
            _tile_dpbf16ps(2, 6, 7);
            _tile_dpbf16ps(3, 6, 7);
            _tile_dpbf16ps(4, 6, 7);
            _tile_dpbf16ps(5, 6, 7);
        }
    });

    _tile_release();
    return 0;
}
