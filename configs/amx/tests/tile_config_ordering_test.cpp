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

// test to check that configuration controls following tile loads do not not
// issue before older loads complete
int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 2;
    config.columnBytes[1] = 8;
    config.rows[1] = 2;

    alignas(64) int32_t source_0[2][2] = {
        {1, 2},
        {3, 4},
    };
    alignas(64) int32_t source_1[2][2] = {
        {5, 6},
        {7, 8},
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source_0, sizeof(source_0[0]));
    amx_tile_loadd(1, source_1, sizeof(source_1[0]));

    // the simuout should show both load issues before either load completes
    // this loadconfig should never issue before a older instrcution
    // this is a temp replacment to tilerelease!
    alignas(64) TileConfig release = {};
    amx_tile_loadconfig(&release);

    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();

    return 0;
}
