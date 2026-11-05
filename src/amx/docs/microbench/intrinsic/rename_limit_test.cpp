//   g++ -O2 -mamx-tile -mamx-bf16 -o rename_limit_test rename_limit_test.cpp
//   AMX_BENCH_CPU=<core> ./rename_limit_test
// /sys/kernel/mm/transparent_hugepage/enabled should be always or madvise.

#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "perf_helpers.h"

const int Iterations = 100;

const size_t Spacing = 1024;
const size_t BufferBytes = size_t(256) << 20;
const size_t Nodes = BufferBytes / Spacing;

alignas(64) TileConfig config;
alignas(64) uint8_t source[1024];
uint8_t *buffer;
uint8_t *chainA;
uint8_t *chainB;

template <int N, bool DotProducts>
inline void
tileWrites()
{
    asm volatile("" ::: "memory");
    for (int j = 0; j < N; ++j) {
        if constexpr (DotProducts) {
            _tile_dpbf16ps(3, 6, 7);
        } else {
            _tile_loadd(3, source, 64);
        }
    }
    asm volatile("" ::: "memory");
}

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

int
main()
{
    setUpBenchmark();

    buffer = static_cast<uint8_t *>(std::aligned_alloc(2 << 20, BufferBytes));
    madvise(buffer, BufferBytes, MADV_HUGEPAGE);
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

    fillConfig(config, 16, 64);
    asm volatile("" ::: "memory");
    _tile_loadconfig(&config);

    // How many tile writes can be in flight before renaming runs out of
    // physical tiles? Each iteration has two independent DRAM loads (chains A
    // and B walk a random cycle through a flushed 256 MiB buffer), each
    // followed by N writes to tile 3, with tile loads or with dot products.
    // While load A waits for DRAM nothing after it can retire, so every tile
    // write behind it needs a new physical tile. If they all get one, load B
    // starts while A is still waiting and an iteration takes about one DRAM
    // miss. Once N needs more physical tiles than are free, B can't start
    // until A is done, and an iteration jumps to about two misses. The N of
    // the jump is the number of spare physical tiles. (Dot products also
    // need 2N x 16 cycles to execute, which hides the jump past N = ~16.)
    for (const Test &test : tests) {
        region(test.name, Iterations, test.run);
    }

    _tile_release();
    std::free(buffer);
    return 0;
}
