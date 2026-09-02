#include <cstdint>
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

enum
{
    M = 16,
    N = 16,
    K = 32,
    A_ROW_BYTES = K * sizeof(uint16_t),
    PACKED_B_ROW_BYTES = N * 2 * sizeof(uint16_t),
    C_ROW_BYTES = N * sizeof(float)
};

int
main()
{
    alignas(64) uint16_t matrix_a[M][K];
    alignas(64) uint16_t matrix_b[K][N];
    alignas(64) uint16_t packed_b[K / 2][2 * N];
    alignas(64) float matrix_c[M][N] = {0};

    // Initialize A and B with BF16 values (0x4000 = 2.0, 0x4080 = 4.0)
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            matrix_a[m][k] = 0x4000;
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            matrix_b[k][n] = 0x4080;
        }
    }

    // Pack matrix B: pairs of adjacent K elements in rows of 2*N
    for (int k = 0; k < K; k += 2) {
        for (int n = 0; n < N; ++n) {
            packed_b[k / 2][2 * n] = matrix_b[k][n];
            packed_b[k / 2][2 * n + 1] = matrix_b[k + 1][n];
        }
    }

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    config.startRow = 0;

    // TMM0: 16x16 FP32 result
    config.rows[0] = M;
    config.columnBytes[0] = C_ROW_BYTES;

    // TMM1: A as 16 rows of 32 BF16 values (64 bytes/row)
    config.rows[1] = M;
    config.columnBytes[1] = A_ROW_BYTES;

    // TMM2: Packed B as 16 rows of 32 BF16 values (64 bytes/row)
    config.rows[2] = K / 2;
    config.columnBytes[2] = PACKED_B_ROW_BYTES;

    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);

    amx_tile_loadconfig(&config);
    amx_tile_zero(0);
    amx_tile_loadd(1, matrix_a, A_ROW_BYTES);
    amx_tile_loadd(2, packed_b, PACKED_B_ROW_BYTES);
    amx_tile_dpbf16ps(0, 1, 2);
    amx_tile_stored(0, matrix_c, C_ROW_BYTES);

    amx_tile_loadconfig(&release);
    m5_work_end(0, 0);

    m5_exit(750);
    m5_quiesce();

    return 0;
}