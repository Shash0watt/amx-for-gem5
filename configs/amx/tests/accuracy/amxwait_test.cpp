#include <cstdint>

#include <gem5/m5ops.h>

#include "tile_config.hh"

int
main()
{
    // TMM0 holds the AMX-stored data, TMM1 holds what the CPU observed.
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 2;
    config.columnBytes[1] = 8;
    config.rows[1] = 2;

    alignas(64) uint16_t source[2][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080}, // 1, 2, 3, 4
        {0x40a0, 0x40c0, 0x40e0, 0x4100}, // 5, 6, 7, 8
    };
    // Already filled with -1.0 so that a CPU read before the AMX store is visible
    alignas(64) uint16_t destination[2][4] = {
        {0xbf80, 0xbf80, 0xbf80, 0xbf80},
        {0xbf80, 0xbf80, 0xbf80, 0xbf80},
    };
    alignas(64) uint16_t snapshot[2][4];

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_tile_stored(0, destination, sizeof(destination[0]));

    // Without amx_wait the copy below reads the old -1.0 (0xbf80) values,
    // because the store is still queued behind the load.
    amx_wait();

    // Copy everything at once, so every value is read at the same point
    // relative to the store.
    snapshot[0][0] = destination[0][0];
    snapshot[0][1] = destination[0][1];
    snapshot[0][2] = destination[0][2];
    snapshot[0][3] = destination[0][3];
    snapshot[1][0] = destination[1][0];
    snapshot[1][1] = destination[1][1];
    snapshot[1][2] = destination[1][2];
    snapshot[1][3] = destination[1][3];

    // Load what the CPU observed into TMM1 so the dump records it. With a
    // working amx_wait, TMM1 matches TMM0.
    amx_tile_loadd(1, snapshot, sizeof(snapshot[0]));
    amx_dump_state("amxwait");

    // Let the dump finish before exiting.
    amx_wait();
    m5_work_end(0, 0);

   
    return 0;
}
