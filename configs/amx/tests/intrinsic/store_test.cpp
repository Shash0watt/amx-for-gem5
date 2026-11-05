// TILESTORED latency and throughput in gem5, data in L1. The hardware
// version is src/amx/docs/microbench/intrinsic/store_test.cpp.
//
// Each test is one work region of Ops instructions. The config prints the
// region's cycles at "--- END of AMX Region ---"; divide by Ops.
//
// Hardware waits for stores with MFENCE; here amx_wait does it, as an AMX
// store only finishes once its writes have reached the cache.

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

const int Ops = 960; // divisible by 8, the loop size below

alignas(64) TileConfig config;
alignas(64) uint8_t destination[8 * 1024]; // eight 1 KiB tiles, fits in L1

// Time test() as one work region. The region starts after all earlier AMX
// work and ends once the test's AMX work has finished.
void
region(const char *name, void (*test)())
{
    amx_wait();
    std::printf("%s\n", name);
    std::fflush(stdout); // print the name before the region marker
    m5_work_begin(0, 0);
    test();
    amx_wait(); // hardware: MFENCE
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

    // Latency: a store's address doesn't come from a tile, so stores can't
    // form a chain. Instead wait for each one to finish; the latency is
    // "store + wait" minus "wait only". The same 1 KiB every time, so it
    // stays in L1.
    region("wait only (baseline)", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_wait(); // hardware: MFENCE
        }
    });
    region("TILESTORED: store + wait, L1", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_stored(0, destination, 64);
            amx_wait(); // hardware: MFENCE
        }
    });

    // Throughput: eight different tiles to eight 1 KiB buffers, so no
    // store waits on another.
    region("TILESTORED: eight tiles, L1", [] {
        for (int i = 0; i < Ops / 8; ++i) {
            amx_tile_stored(0, destination + 0 * 1024, 64);
            amx_tile_stored(1, destination + 1 * 1024, 64);
            amx_tile_stored(2, destination + 2 * 1024, 64);
            amx_tile_stored(3, destination + 3 * 1024, 64);
            amx_tile_stored(4, destination + 4 * 1024, 64);
            amx_tile_stored(5, destination + 5 * 1024, 64);
            amx_tile_stored(6, destination + 6 * 1024, 64);
            amx_tile_stored(7, destination + 7 * 1024, 64);
        }
    });

    return 0;
}
