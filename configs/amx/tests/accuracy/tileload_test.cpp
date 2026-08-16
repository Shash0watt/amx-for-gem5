#include <cstdint>

#include <gem5/m5ops.h>

#include "tile_config.hh"

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 2;

    // BF16 values:
    //   row 0 = 1, 2, 3, 4
    //   row 1 = 5, 6, 7, 8
    alignas(64) uint16_t source[2][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080},
        {0x40a0, 0x40c0, 0x40e0, 0x4100},
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_dump_state("tileload");
    m5_work_end(0, 0);

    m5_exit(1000);
    m5_quiesce();
    return 0;
}
