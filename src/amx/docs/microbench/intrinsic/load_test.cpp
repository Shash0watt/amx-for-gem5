//   g++ -O2 -mamx-tile -mamx-bf16 -o load_test load_test.cpp
//   AMX_BENCH_CPU=<core> ./load_test
// For the DRAM test, /sys/kernel/mm/transparent_hugepage/enabled should be
// always or madvise.

#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>

#include "perf_helpers.h"

const int Ops = 960;

const size_t Stride = 4096;
const int L2Tiles = 256;
const size_t DramBytes = size_t(256) << 20;

alignas(64) TileConfig config;
alignas(64) uint8_t source[8 * 1024];
alignas(4096) uint8_t l2Source[L2Tiles * Stride];
uint8_t *dram;
size_t dramOffset = 0;

int
main()
{
    setUpBenchmark();

    dram = static_cast<uint8_t *>(std::aligned_alloc(2 << 20, DramBytes));
    madvise(dram, DramBytes, MADV_HUGEPAGE);
    for (size_t offset = 0; offset < DramBytes; offset += Stride) {
        dram[offset + Stride / 2] = 0;
    }

    fillConfig(config, 16, 64);
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // Latency from L1, L2 and DRAM: a load's address doesn't come from a
    // tile, so loads can't be chained. Instead an LFENCE after each one
    // waits for it; latency = "load + LFENCE" minus "LFENCE only". L1 loads
    // the same 1 KiB every time. L2 cycles through 256 tiles (256 KiB, more
    // than L1's 48 KiB, less than L2's 2 MiB). DRAM loads a new tile from a
    // 256 MiB buffer each time, on 2 MiB pages so page walks don't add to
    // it. L2 and DRAM tiles are one per 4 KiB, so prefetchers don't help.
    region("LFENCE only (baseline)", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _mm_lfence();
        }
    });
    region("load + LFENCE, L1", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_loadd(0, source, 64);
            _mm_lfence();
        }
    });
    region("load + LFENCE, L2", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_loadd(0, l2Source + (i % L2Tiles) * Stride, 64);
            _mm_lfence();
        }
    });
    region("load + LFENCE, DRAM", Ops, [] {
        for (int i = 0; i < Ops; ++i) {
            _tile_loadd(0, dram + dramOffset, 64);
            dramOffset = (dramOffset + Stride) % DramBytes;
            _mm_lfence();
        }
    });

    // Throughput: loads of eight different tiles from eight 1 KiB buffers
    // back to back, so none waits for another and cycles per op = the issue
    // interval.
    region("eight tiles, L1", Ops, [] {
        for (int i = 0; i < Ops / 8; ++i) {
            _tile_loadd(0, source + 0 * 1024, 64);
            _tile_loadd(1, source + 1 * 1024, 64);
            _tile_loadd(2, source + 2 * 1024, 64);
            _tile_loadd(3, source + 3 * 1024, 64);
            _tile_loadd(4, source + 4 * 1024, 64);
            _tile_loadd(5, source + 5 * 1024, 64);
            _tile_loadd(6, source + 6 * 1024, 64);
            _tile_loadd(7, source + 7 * 1024, 64);
        }
    });

    _tile_release();
    std::free(dram);
    return 0;
}
