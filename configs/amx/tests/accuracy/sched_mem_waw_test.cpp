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

// Scheduler rule: memory WAW, two stores to the same bytes.
//
// The first store waits ~45 cycles for TMM0 to load; the second store's
// source (TILEZERO TMM1) is ready at cycle 16. Only the memory-overlap check
// keeps the second store behind the first. Without it the zeros are written
// first and then overwritten, leaving 1..4 instead of 0s in `buffer`.
int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 1;
    config.columnBytes[1] = 8;
    config.rows[1] = 1;

    alignas(64) uint16_t source[1][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080}, // 1, 2, 3, 4
    };
    alignas(64) uint16_t buffer[1][4] = {
        {0xbf80, 0xbf80, 0xbf80, 0xbf80}, // -1
    };

    // Reload `buffer` into TMM2 for inspection.
    alignas(64) TileConfig inspectConfig = {};
    inspectConfig.paletteId = 1;
    inspectConfig.columnBytes[2] = 8;
    inspectConfig.rows[2] = 1;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_tile_stored(0, buffer, sizeof(buffer[0]));
    amx_tile_zero(1);
    amx_tile_stored(1, buffer, sizeof(buffer[0]));

    // Configuration is a barrier, so both stores complete before the reload.
    amx_tile_loadconfig(&inspectConfig);
    amx_tile_loadd(2, buffer, sizeof(buffer[0]));
    amx_dump_state("sched_mem_waw");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
