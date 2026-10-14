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

// Scheduler rule: memory WAR, a store after a load from the same bytes.
//
// Three filler loads occupy the load pipeline, so the TMM1 load from
// `buffer` cannot issue until cycle 24. The store of TMM0 (zeroed by the
// configuration) has no tile dependency and a free store pipeline, so only
// the memory-overlap check keeps it behind that load. Without that check
// the store writes zeros at cycle 0 and TMM1 reads zeros instead of 1..4.
int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 5; ++tile) {
        config.columnBytes[tile] = 8;
        config.rows[tile] = 1;
    }

    alignas(64) uint16_t filler[1][4] = {
        {0x40a0, 0x40c0, 0x40e0, 0x4100}, // 5, 6, 7, 8
    };
    alignas(64) uint16_t buffer[1][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080}, // 1, 2, 3, 4
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(2, filler, sizeof(filler[0]));
    amx_tile_loadd(3, filler, sizeof(filler[0]));
    amx_tile_loadd(4, filler, sizeof(filler[0]));
    amx_tile_loadd(1, buffer, sizeof(buffer[0]));
    amx_tile_stored(0, buffer, sizeof(buffer[0]));
    amx_dump_state("sched_mem_war");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
