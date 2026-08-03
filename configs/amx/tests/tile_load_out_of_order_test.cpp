#include <cstdint>
#include <iostream>

#include <gem5/m5ops.h>

struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t columnBytes[16];
    uint8_t rows[16];
};

static_assert(sizeof(TileConfig) == 64);

int
main()
{
    // 16x16 arrays of int32_t
    // 16 * 16 * sizeof(int32_t) = 1024 bytes (1KB)
    // aligned to 64 bytes to ensure cache line alignment
    alignas(64) int32_t array1[16][16];
    alignas(64) int32_t array2[16][16];
    size_t stride = 16 * sizeof(int32_t);

    // initialize arrays with some values
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            array1[i][j] = i * 16 + j;
            array2[i][j] = (15 - i) * 16 + (15 - j);
        }
    }

    std::cout << "out-of-order tile load test" << std::endl;

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.columnBytes[0] = 64;
    config.rows[0] = 16;
    config.columnBytes[1] = 64;
    config.rows[1] = 16;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);

    // load array1 into tile 0
    amx_tile_loadd(0, array1, stride);

    // load array2 into tile 0 again
    amx_tile_loadd(0, array2, stride);

    // load array1 into tile 1
    amx_tile_loadd(1, array1, stride);
    // we should expect to see this load complete earlier if out of order
    // behaviour is allowed

    alignas(64) TileConfig release = {};
    amx_tile_loadconfig(&release);

    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();

    std::cout << "out-of-order tile load test complete" << std::endl;

    return 0;
}
