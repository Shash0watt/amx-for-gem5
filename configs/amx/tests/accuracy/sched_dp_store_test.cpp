#include <cstdint>

#include <gem5/m5ops.h>

// The 64-byte memory layout consumed by AMX_TILE_LOADCONFIG.
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

// Scheduler rule: tile RAW from a dot product into a store.
//
// The store reads TMM0, which the dot product writes 52 cycles after it
// issues. Stores have no minimum latency, so without the RAW check the store
// writes the old TMM0 (0 or 1) to memory instead of the result 24.
int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 3; ++tile) {
        config.columnBytes[tile] = 4;
        config.rows[tile] = 1;
    }

    alignas(64) float accumulator[1][1] = {{1.0F}};
    alignas(64) uint16_t left[1][2] = {
        {0x4000, 0x4040}, // 2, 3
    };
    alignas(64) uint16_t right[1][2] = {
        {0x4080, 0x40a0}, // 4, 5
    };
    alignas(64) float result[1][1] = {{-1.0F}};

    // Reload the stored FP32 result into TMM3 for inspection.
    alignas(64) TileConfig inspectConfig = {};
    inspectConfig.paletteId = 1;
    inspectConfig.columnBytes[3] = 4;
    inspectConfig.rows[3] = 1;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, accumulator, sizeof(accumulator[0]));
    amx_tile_loadd(1, left, sizeof(left[0]));
    amx_tile_loadd(2, right, sizeof(right[0]));

    // TMM0 = 1 + (2 * 4) + (3 * 5) = 24.
    amx_tile_dpbf16ps(0, 1, 2);
    amx_tile_stored(0, result, sizeof(result[0]));

    // Configuration is a barrier, so the store completes before the reload.
    amx_tile_loadconfig(&inspectConfig);
    amx_tile_loadd(3, result, sizeof(result[0]));
    amx_dump_state("sched_dp_store");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
