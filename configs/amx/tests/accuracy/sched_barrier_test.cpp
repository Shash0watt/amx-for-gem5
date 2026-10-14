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

// Scheduler rule: younger work cannot pass a configuration barrier.
//
// The second configuration reshapes TMM0 to 2x16 bytes. If the final load
// slips past it, the load issues under the old 1x8 shape (~cycle 45) and the
// new configuration then commits (~cycle 249), clearing TMM0 to zeros.
int
main()
{
    alignas(64) TileConfig smallConfig = {};
    smallConfig.paletteId = 1;
    smallConfig.columnBytes[0] = 8;
    smallConfig.rows[0] = 1;

    alignas(64) TileConfig largeConfig = {};
    largeConfig.paletteId = 1;
    largeConfig.columnBytes[0] = 16;
    largeConfig.rows[0] = 2;

    alignas(64) uint16_t small[1][4] = {
        {0x4110, 0x4110, 0x4110, 0x4110}, // 9
    };
    alignas(64) uint16_t large[2][8] = {
        {0x3f80, 0x4000, 0x4040, 0x4080,
         0x40a0, 0x40c0, 0x40e0, 0x4100}, // 1..8
        {0x4110, 0x4120, 0x4130, 0x4140,
         0x4150, 0x4160, 0x4170, 0x4180}, // 9..16
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&smallConfig);
    amx_tile_loadd(0, small, sizeof(small[0]));
    amx_tile_loadconfig(&largeConfig);
    amx_tile_loadd(0, large, sizeof(large[0]));
    amx_dump_state("sched_barrier");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
