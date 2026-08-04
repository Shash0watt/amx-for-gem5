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
    constexpr uint64_t resultTile = 0;
    constexpr uint64_t leftTile = 1;
    constexpr uint64_t rightTile = 2;
    constexpr uint64_t reloadedTile = 3;

    // One FP32 value and one pair of BF16 values fit in a four-byte tile row.
    alignas(64) float accumulator[1][1] = {{1.0F}};
    alignas(64) uint16_t left[1][2] = {{0x4000, 0x4040}};  // 2.0, 3.0
    alignas(64) uint16_t right[1][2] = {{0x4080, 0x40a0}}; // 4.0, 5.0
    alignas(64) float storedResult[1][1] = {};

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (uint64_t tile = resultTile; tile <= reloadedTile; ++tile) {
        config.columnBytes[tile] = sizeof(float);
        config.rows[tile] = 1;
    }

    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);

    amx_tile_loadconfig(&config);
    amx_tile_loadd(resultTile, accumulator, sizeof(accumulator[0]));
    amx_tile_loadd(leftTile, left, sizeof(left[0]));
    amx_tile_loadd(rightTile, right, sizeof(right[0]));

    // TMM0 = 1 + (2 * 4) + (3 * 5) = 24.
    amx_tile_dpbf16ps(resultTile, leftTile, rightTile);
    amx_tile_stored(resultTile, storedResult, sizeof(storedResult[0]));

    // Reloading the configuration waits for the store to reach memory. It
    // also clears the tile registers, so TMM3 can only receive the stored 24.
    amx_tile_loadconfig(&config);
    amx_tile_loadd(reloadedTile, storedResult, sizeof(storedResult[0]));
    amx_tile_loadconfig(&release);

    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();

    return 0;
}
