#include <gem5/m5ops.h>

#include "tile_config.hh"

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 2;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_dump_state("loadconfig");
    amx_wait();
    m5_work_end(0, 0);

   
    return 0;
}
