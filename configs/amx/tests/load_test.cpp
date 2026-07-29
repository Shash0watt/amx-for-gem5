#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <gem5/m5ops.h> // use our m5ops

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
main(int argc, char *argv[])
{
    uint64_t dest_tile = 0;
    size_t stride = 64;

    // set this value to 0 to have the data be perfectly aligned
    int offset = 60;

    // create a matrix that  dosen't fit in the cache-line boundaries
    // intentionally
    alignas(64) int8_t buffer[16 * 64 + 64] = {0};
    // get a pointer to the start of the test location
    int8_t *offset_matrix = &buffer[offset];
    // fil the matrix with incrementing numbers
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 64; ++c) {
            offset_matrix[(r * 64) + c] = c;
        }
    }

    alignas(64) __tilecfg config = {};
    config.palette_id = 1;
    config.colsb[0] = 64;
    config.rows[0] = 16;

    m5_work_begin(0, 0);
 
    // get permission from OS to use intel AMX

    // use replacment op for tile config
    amx_tile_loadconfig(&config);

    // use the replacement m5op for a tile laod
    amx_tile_loadd(dest_tile, offset_matrix, stride);

    alignas(64) __tilecfg release = {};
    amx_tile_loadconfig(&release);

    m5_quiesce_cycle(10000);
    m5_work_end(0, 0);
 
    return 0;
}
