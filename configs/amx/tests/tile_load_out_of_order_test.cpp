#include <cstdint>
#include <gem5/m5ops.h>
#include <iostream>

typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

static_assert(sizeof(__tilecfg) == 64);

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

    alignas(64) __tilecfg config = {};
    config.palette_id = 1;
    config.colsb[0] = 64;
    config.rows[0] = 16;
    config.colsb[1] = 64;
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

    alignas(64) __tilecfg release = {};
    amx_tile_loadconfig(&release);

    m5_quiesce_cycle(10000);
    m5_work_end(0, 0);

    std::cout << "out-of-order tile load test complete" << std::endl;

    return 0;
}
