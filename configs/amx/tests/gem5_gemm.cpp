#include <cstdint>
#include <vector>

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
    constexpr int max_rows = 16;
    constexpr int max_bytes_per_row = 64;

    constexpr int total_rows = 64;
    constexpr int total_cols = 64;

    constexpr int tile_m = max_rows;
    constexpr int tile_k = max_bytes_per_row;
    constexpr int tile_n = max_bytes_per_row / 4;

    std::vector<uint16_t> matA(total_rows * total_cols, 0x4000);
    std::vector<uint16_t> matB(total_rows * total_cols, 0x4080);
    std::vector<float> matC(total_rows * total_cols, 0.0F);

    size_t stride_a = total_cols * sizeof(uint16_t);
    size_t stride_b = total_cols * sizeof(uint16_t);
    size_t stride_c = total_cols * sizeof(float);

    alignas(64) TileConfig config = {};
    config.paletteId = 1;

    config.rows[0] = tile_m;
    config.columnBytes[0] = tile_n * sizeof(float);

    config.rows[1] = tile_m;
    config.columnBytes[1] = tile_k;

    config.rows[2] = tile_k / 4;
    config.columnBytes[2] = tile_n * 4;

    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);

    for (int m = 0; m < total_rows; m += tile_m) {
        for (int n = 0; n < total_cols; n += tile_n) {
            amx_tile_zero(0);

            for (int k = 0; k < total_cols; k += tile_k / sizeof(uint16_t)) {
                const uint16_t *ptr_a = &matA[m * total_cols + k];
                const uint16_t *ptr_b = &matB[k * total_cols + n];

                amx_tile_loadd(1, ptr_a, stride_a);
                amx_tile_loadd(2, ptr_b, stride_b);
                amx_tile_dpbf16ps(0, 1, 2);
            }

            float *ptr_c = &matC[m * total_cols + n];
            amx_tile_stored(0, ptr_c, stride_c);
        }
    }
    m5_work_end(0, 0);

    amx_tile_loadconfig(&release);
    
    m5_exit(750);
    m5_quiesce();

    return 0;
}

