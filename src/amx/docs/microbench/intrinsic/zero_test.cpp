//   g++ -O2 -mamx-tile -mamx-bf16 -o zero_test zero_test.cpp
//   AMX_BENCH_CPU=<core> ./zero_test

#include "perf_helpers.h"

const int Ops = 960;

alignas(64) TileConfig config;

int
main()
{
    setUpBenchmark();

    fillConfig(config, 16, 64);
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // Latency: a zero doesn't read a tile, so zeros can't be chained.
    // Instead an LFENCE after each one waits for it; latency = "zero +
    // wait" minus "wait only".
    region("wait only (baseline)", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _mm_lfence();
        }
    });
    region("TILEZERO: zero + wait", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_zero(0);
            _mm_lfence();
        }
    });

    // Zero-to-use, how long until a dot product can read the zeroed tile:
    // "zero + TDPBF16PS + LFENCE" minus "TDPBF16PS + LFENCE", zeroing the
    // dot product's A and its C.
    region("TDPBF16PS + LFENCE (baseline)", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_dpbf16ps(0, 6, 7);
            _mm_lfence();
        }
    });
    region("zero A + TDPBF16PS + LFENCE", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_zero(1);
            _tile_dpbf16ps(0, 1, 7);
            _mm_lfence();
        }
    });
    region("zero C + TDPBF16PS + LFENCE", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_zero(0);
            _tile_dpbf16ps(0, 6, 7);
            _mm_lfence();
        }
    });

    // Throughput: zeros to eight different tiles back to back, so none
    // waits for another and cycles per op = the issue interval.
    region("TILEZERO: eight tiles", Ops, [] {
        for (int i = 0; i < Ops / 8; ++i) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);
        }
    });

    _tile_release();
    return 0;
}
