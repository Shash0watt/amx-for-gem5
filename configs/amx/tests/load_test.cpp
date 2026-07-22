#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <gem5/m5ops.h> // use our m5ops

int
main(int argc, char *argv[])
{
    uint64_t dest_tile = 0;
    size_t stride = 64;

    // set this value to 0 to have the data be perfectly aligned
    int offset = 60;

    // creates a matrix that dosen't fit in the cache intentionally
    alignas(64) int8_t buffer[16 * 64 + 64] = {0};
    // get a pointer to the start of the test location
    int8_t *offset_matrix = &buffer[offset];
    // fil the matrix with incrementing numbers
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 64; ++c) {
            offset_matrix[(r * 64) + c] = c;
        }
    }

    m5_work_begin(0, 0);

    // get permission from OS to use intel AMX

    // load the tile
    // _tile_loadconfig(&tile_config_struct);

    // use the replacement m5op for a tile laod
    amx_tile_loadd(dest_tile, offset_matrix, stride);

    m5_work_end(0, 0);

    return 0;
}