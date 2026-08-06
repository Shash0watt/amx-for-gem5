#include <cstdint>

#include <gem5/m5ops.h>

struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t columnBytes[16];
    uint8_t rows[16];
};

static_assert(sizeof(TileConfig) == 64);

int
main()
{
    // One FP32 destination element and one pair of BF16 inputs are enough to
    // exercise the complete TDPBF16PS timing and execution path.
    alignas(64) float dest_tile[1][1] = {{1.0F}};
    alignas(64) uint16_t tile_a[1][2] = {{0x4000, 0x4040}}; // 2.0, 3.0
    alignas(64) uint16_t tile_b[1][2] = {{0x4080, 0x40a0}}; // 4.0, 5.0

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 3; ++tile) {
        config.columnBytes[tile] = sizeof(uint32_t);
        config.rows[tile] = 1;
    }

    alignas(64) TileConfig release = {};

    // warmup
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, dest_tile, sizeof(dest_tile[0]));
    amx_tile_loadd(1, tile_a, sizeof(tile_a[0]));
    amx_tile_loadd(2, tile_b, sizeof(tile_b[0]));

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, dest_tile, sizeof(dest_tile[0]));
    amx_tile_loadd(1, tile_a, sizeof(tile_a[0]));
    amx_tile_loadd(2, tile_b, sizeof(tile_b[0]));

    // Expected completion value:
    //   1.0 + (2.0 * 4.0) + (3.0 * 5.0) = 24.0
    // The AMX trace should publish that value exactly 52 cycles after issue.
    amx_tile_dpbf16ps(0, 1, 2);

    amx_tile_loadconfig(&release);
    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();
    return 0;
}
