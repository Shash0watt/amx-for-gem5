#include <cstdint>

#include <gem5/m5ops.h>

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

static_assert(sizeof(TileConfig) == 64);

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 4;
    config.rows[0] = 1;
    config.columnBytes[1] = 4;
    config.rows[1] = 1;

    alignas(64) uint16_t tile0[2] = {0x3f80, 0x4000}; // 1, 2
    alignas(64) uint16_t tile1[2] = {0x4040, 0x4080}; // 3, 4
    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(1, tile1, sizeof(tile1));
    amx_dump_state("initial_state");

    amx_tile_loadd(0, tile0, sizeof(tile0));
    amx_dump_state("before_young");

    // TMM1 is independent of the older TMM0 load. It would pass that load if
    // the preceding dump did not also block younger instructions.
    amx_tile_zero(1);
    amx_dump_state("after_young");

    amx_tile_loadconfig(&release);
    amx_dump_state("released");
    m5_work_end(0, 0);

    m5_exit(1000);
    m5_quiesce();
    return 0;
}
