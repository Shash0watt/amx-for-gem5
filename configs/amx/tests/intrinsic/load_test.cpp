// TILELOADD latency and throughput in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/load_test.cpp, which explains the tests.
//
// Each test is one work region of Ops instructions. The config prints the
// region's cycles at "--- END of AMX Region ---"; divide by Ops.

#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

const size_t Stride = 4096;
const int L2Tiles = 256; // 256 KiB of tiles: misses L1 (48 KiB), fits L2 (2 MiB)
const size_t DramBytes = size_t(256) << 20; // more than L3

alignas(64) TileConfig config;
alignas(64) uint8_t source[8 * 1024]; // eight 1 KiB tiles, 8 KiB fits in L1
alignas(4096) uint8_t l2Source[L2Tiles * Stride];
uint8_t *dram;
size_t dramOffset = 0; // moves on every load, so no DRAM tile is reused

// Time test() as one work region. The region starts after all earlier AMX
// work and ends once the test's AMX work has finished. test() runs once
// untimed first, like on hardware, so L1 and L2 data is in the cache.
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
    // Create the DRAM pages by writing to each, away from the tile at its
    // start, so the tiles themselves aren't brought into the caches.
    dram = static_cast<uint8_t *>(std::aligned_alloc(2 << 20, DramBytes));
    madvise(dram, DramBytes, MADV_HUGEPAGE); // ignored by gem5
    for (size_t offset = 0; offset < DramBytes; offset += Stride) {
        dram[offset + Stride / 2] = 0;
    }

    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.columnBytes[tile] = 64;
    }
    amx_tile_loadconfig(&config);

    // Latency, from L1 (the same 1 KiB every time), L2 and DRAM.
    region("LFENCE only (baseline)", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_wait(); // hardware: LFENCE
        }
    });
    region("load + LFENCE, L1", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_loadd(0, source, 64);
            amx_wait(); // hardware: LFENCE
        }
    });
    region("load + LFENCE, L2", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_loadd(0, l2Source + (i % L2Tiles) * Stride, 64);
            amx_wait(); // hardware: LFENCE
        }
    });
    region("load + LFENCE, DRAM", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_loadd(0, dram + dramOffset, 64);
            dramOffset = (dramOffset + Stride) % DramBytes;
            amx_wait(); // hardware: LFENCE
        }
    });

    // Throughput: eight different tiles from eight 1 KiB buffers.
    region("eight tiles, L1", [] {
        for (int i = 0; i < Ops / 8; ++i) {
            amx_tile_loadd(0, source + 0 * 1024, 64);
            amx_tile_loadd(1, source + 1 * 1024, 64);
            amx_tile_loadd(2, source + 2 * 1024, 64);
            amx_tile_loadd(3, source + 3 * 1024, 64);
            amx_tile_loadd(4, source + 4 * 1024, 64);
            amx_tile_loadd(5, source + 5 * 1024, 64);
            amx_tile_loadd(6, source + 6 * 1024, 64);
            amx_tile_loadd(7, source + 7 * 1024, 64);
        }
    });

    std::free(dram);
    return 0;
}
