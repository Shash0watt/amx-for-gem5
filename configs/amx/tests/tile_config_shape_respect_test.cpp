#include <cstddef>
#include <cstdint>

#include <gem5/m5ops.h>

struct TileConfig
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static_assert(sizeof(TileConfig) == 64);

int
main()
{
    // TILELOADD receives only an address and a stride, so an undersized C
    // array would be an out-of-bounds read rather than an architectural shape
    // check.
    alignas(64) int32_t source[16][16] = {};
    for (int row = 0; row < 16; ++row) {
        for (int column = 0; column < 16; ++column) {
            source[row][column] = row * 100 + column;
        }
    }

    alignas(64) TileConfig full = {};
    full.palette_id = 1;
    full.colsb[0] = 64;
    full.rows[0] = 16;

    alignas(64) TileConfig small = {};
    small.palette_id = 1;
    small.colsb[0] = 16;
    small.rows[0] = 4;

    alignas(64) TileConfig release = {};

    constexpr uint64_t tile = 0;
    constexpr size_t stride = sizeof(source[0]);

    m5_work_begin(0, 0);

    amx_tile_loadconfig(&full);
    amx_tile_loadd(tile, source, stride);

    // Configuration operations are queue barriers. The first load must finish
    // before this commits, and the otherwise-identical second load must use
    // this smaller shape.
    amx_tile_loadconfig(&small);
    amx_tile_loadd(tile, source, stride);

    amx_tile_loadconfig(&release);
    
    m5_quiesce_cycle(10);
    m5_work_end(0, 0);

    return 0;
}
