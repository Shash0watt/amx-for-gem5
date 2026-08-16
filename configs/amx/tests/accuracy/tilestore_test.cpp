#include <cstdint>

#include <gem5/m5ops.h>

#include "tile_config.hh"

int
main()
{
    alignas(64) TileConfig storeConfig = {};
    storeConfig.paletteId = 1;
    storeConfig.columnBytes[0] = 8;
    storeConfig.rows[0] = 2;

    // Store four BF16 values into each eight-value destination row.
    alignas(64) uint16_t source[2][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080}, // 1, 2, 3, 4
        {0x40a0, 0x40c0, 0x40e0, 0x4100}, // 5, 6, 7, 8
    };
    alignas(64) uint16_t destination[2][8] = {
        {0xbf80, 0xbf80, 0xbf80, 0xbf80,
         0xbf80, 0xbf80, 0xbf80, 0xbf80},
        {0xbf80, 0xbf80, 0xbf80, 0xbf80,
         0xbf80, 0xbf80, 0xbf80, 0xbf80},
    };

    // Reload the complete destination rows into TMM1 for inspection.
    alignas(64) TileConfig inspectConfig = {};
    inspectConfig.paletteId = 1;
    inspectConfig.columnBytes[1] = 16;
    inspectConfig.rows[1] = 2;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&storeConfig);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_tile_stored(0, destination, sizeof(destination[0]));

    // Configuration is a barrier, so the store completes before the reload.
    amx_tile_loadconfig(&inspectConfig);
    amx_tile_loadd(1, destination, sizeof(destination[0]));
    amx_dump_state("tilestore");
    m5_work_end(0, 0);

    m5_exit(1000);
    m5_quiesce();
    return 0;
}
