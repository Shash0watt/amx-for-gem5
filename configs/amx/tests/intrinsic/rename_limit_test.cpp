// How many tile writes can be in flight before renaming runs out of physical
// tiles, in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/rename_limit_test.cpp, which explains it.
//
// Each test is one work region of Iterations iterations. The config prints
// the region's cycles at "--- END of AMX Region ---"; divide by Iterations.

#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

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

const int Iterations = 100;

// One chain node per 1 KiB of a 256 MiB buffer, visited in a random order.
// The two chains start half the cycle apart, so no node is visited twice.
const size_t Spacing = 1024;
const size_t BufferBytes = size_t(256) << 20;
const size_t Nodes = BufferBytes / Spacing;

alignas(64) TileConfig config;
alignas(64) uint8_t source[1024];
uint8_t *buffer;
uint8_t *chainA;
uint8_t *chainB;

// N writes to tile 3: tile loads, or dot products accumulating into it.
template <int N, bool DotProducts>
inline void
tileWrites()
{
    asm volatile("" ::: "memory");
    for (int j = 0; j < N; ++j) {
        if constexpr (DotProducts) {
            amx_tile_dpbf16ps(3, 6, 7);
        } else {
            amx_tile_loadd(3, source, 64);
        }
    }
    asm volatile("" ::: "memory");
}

// Two independent DRAM loads per iteration, each followed by N tile writes.
template <int N, bool DotProducts>
void
step()
{
    uint8_t *a = chainA;
    uint8_t *b = chainB;
    for (int i = 0; i < Iterations; ++i) {
        a = *reinterpret_cast<uint8_t *volatile *>(a);
        tileWrites<N, DotProducts>();
        b = *reinterpret_cast<uint8_t *volatile *>(b);
        tileWrites<N, DotProducts>();
    }
    chainA = a;
    chainB = b;
}

struct Test
{
    const char *name;
    void (*run)();
};

const Test tests[] = {
    {"tile loads, N = 0", step<0, false>},
    {"tile loads, N = 2", step<2, false>},
    {"tile loads, N = 4", step<4, false>},
    {"tile loads, N = 6", step<6, false>},
    {"tile loads, N = 8", step<8, false>},
    {"tile loads, N = 10", step<10, false>},
    {"tile loads, N = 12", step<12, false>},
    {"tile loads, N = 14", step<14, false>},
    {"tile loads, N = 16", step<16, false>},
    {"tile loads, N = 18", step<18, false>},
    {"tile loads, N = 20", step<20, false>},
    {"tile loads, N = 22", step<22, false>},
    {"tile loads, N = 24", step<24, false>},
    {"dot products, N = 0", step<0, true>},
    {"dot products, N = 2", step<2, true>},
    {"dot products, N = 4", step<4, true>},
    {"dot products, N = 6", step<6, true>},
    {"dot products, N = 8", step<8, true>},
    {"dot products, N = 10", step<10, true>},
    {"dot products, N = 12", step<12, true>},
    {"dot products, N = 14", step<14, true>},
    {"dot products, N = 16", step<16, true>},
    {"dot products, N = 18", step<18, true>},
    {"dot products, N = 20", step<20, true>},
    {"dot products, N = 22", step<22, true>},
    {"dot products, N = 24", step<24, true>},
};

// Time test() as one work region. The region starts after all earlier AMX
// work and ends once the test's AMX work has finished. test() runs once
// untimed first, like on hardware.
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
    // Link the nodes into one random cycle, then flush them all so every
    // chain load misses to DRAM.
    buffer = static_cast<uint8_t *>(std::aligned_alloc(2 << 20, BufferBytes));
    madvise(buffer, BufferBytes, MADV_HUGEPAGE); // ignored by gem5
    std::vector<uint32_t> order(Nodes);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(1));
    for (size_t i = 0; i < Nodes; ++i) {
        *reinterpret_cast<uint8_t **>(buffer + order[i] * Spacing) =
            buffer + order[(i + 1) % Nodes] * Spacing;
    }
    chainA = buffer + order[0] * Spacing;
    chainB = buffer + order[Nodes / 2] * Spacing;
    for (size_t i = 0; i < Nodes; ++i) {
        _mm_clflush(buffer + i * Spacing);
    }
    _mm_mfence();

    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.columnBytes[tile] = 64;
    }
    amx_tile_loadconfig(&config);

    for (const Test &test : tests) {
        region(test.name, test.run);
    }

    std::free(buffer);
    return 0;
}
