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

// Scheduler rule: memory RAW, a load after a store to the same bytes.
//
// The store waits ~45 cycles for TMM0 to load. The TMM1 load has no tile
// dependency, so only the memory-overlap check keeps it behind the store.
// Without that check it reads `buffer` at cycle 8 and sees -1s.
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

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_tile_stored(0, buffer, sizeof(buffer[0]));
    amx_tile_loadd(1, buffer, sizeof(buffer[0]));
    amx_dump_state("sched_mem_raw");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
