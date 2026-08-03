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
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = sizeof(int32_t);
    config.rows[0] = 1;
    config.columnBytes[1] = sizeof(int32_t);
    config.rows[1] = 1;

    alignas(64) int32_t tile_a[1][1] = {{11}};
    alignas(64) int32_t tile_b[1][1] = {{22}};
    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);

    // These independent loads should issue eight cycles apart. Each should
    // remain in flight until both memory and its 45-cycle latency have elapsed.
    amx_tile_loadd(0, tile_a, sizeof(tile_a[0]));
    amx_tile_loadd(1, tile_b, sizeof(tile_b[0]));

    // The release is also a convenient queue barrier: its configuration load
    // cannot issue until both older tile writes have completed.
    amx_tile_loadconfig(&release);
    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();
    return 0;
}
