// LDTILECFG latency and throughput in gem5. The hardware version is
// src/amx/docs/microbench/intrinsic/config_test.cpp.
//
// Each test is one work region of Ops instructions. The config prints the
// region's cycles at "--- END of AMX Region ---"; divide by Ops.

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

const int Ops = 960; // divisible by 2, the loop size below

alignas(64) TileConfig fullTiles; // 16 rows x 64 bytes
alignas(64) TileConfig halfTiles; // 8 rows x 32 bytes

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
    amx_wait(); // hardware: LFENCE
    m5_work_end(0, 0);
}

void
fillConfig(TileConfig &config, int rows, int columnBytes)
{
    config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = rows;
        config.columnBytes[tile] = columnBytes;
    }
}

int
main()
{
    fillConfig(fullTiles, 16, 64);
    fillConfig(halfTiles, 8, 32);

    // Latency: a config doesn't read a tile, so configs can't form a chain.
    // Instead wait for each one to finish; the latency is "config + wait"
    // minus "wait only".
    region("wait only (baseline)", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_wait(); // hardware: LFENCE
        }
    });
    region("LDTILECFG: config + wait", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_loadconfig(&fullTiles);
            amx_wait(); // hardware: LFENCE
        }
    });

    // Throughput: back to back, the same config and two different configs
    // in turn (in case loading an unchanged config is cheaper).
    region("LDTILECFG: same config", [] {
        for (int i = 0; i < Ops; ++i) {
            amx_tile_loadconfig(&fullTiles);
        }
    });
    region("LDTILECFG: alternating configs", [] {
        for (int i = 0; i < Ops / 2; ++i) {
            amx_tile_loadconfig(&fullTiles);
            amx_tile_loadconfig(&halfTiles);
        }
    });

    return 0;
}
