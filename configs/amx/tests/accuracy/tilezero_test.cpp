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

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 8;
    config.rows[0] = 1;

    // Load nonzero BF16 data first so TILEZERO has an observable effect.
    alignas(64) uint16_t source[1][4] = {
        {0x3f80, 0x4000, 0x4040, 0x4080}, // 1, 2, 3, 4
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source, sizeof(source[0]));
    amx_tile_zero(0);
    amx_dump_state("tilezero");
    amx_wait();
    m5_work_end(0, 0);

   
    return 0;
}
