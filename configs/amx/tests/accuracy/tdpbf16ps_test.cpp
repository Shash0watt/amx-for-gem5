#include <cstdint>

#include <gem5/m5ops.h>

#include "tile_config.hh"

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 3; ++tile) {
        config.columnBytes[tile] = 4;
        config.rows[tile] = 1;
    }

    alignas(64) float accumulator[1][1] = {{1.0F}};
    alignas(64) uint16_t left[1][2] = {
        {0x4000, 0x4040}, // 2, 3
    };
    alignas(64) uint16_t right[1][2] = {
        {0x4080, 0x40a0}, // 4, 5
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, accumulator, sizeof(accumulator[0]));
    amx_tile_loadd(1, left, sizeof(left[0]));
    amx_tile_loadd(2, right, sizeof(right[0]));

    // TMM0 = 1 + (2 * 4) + (3 * 5) = 24.
    amx_tile_dpbf16ps(0, 1, 2);
    amx_dump_state("tdpbf16ps");
    m5_work_end(0, 0);

    m5_exit(1000);
    m5_quiesce();
    return 0;
}
