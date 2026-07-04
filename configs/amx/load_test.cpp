#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <gem5/m5ops.h> // for gem5 magic ops

int
main()
{
    uint64_t dest_tile = 0;
    size_t stride = 64;

    // make a 16x64 matrix grid.
    // alignas(64) guarantees that the first row address starts at the begining
    // of a cache line because my simobject cannot handle it yet
    alignas(64) int8_t matrix[16][64] = {0};

    // fill the matrix with distinct hex patterns
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 64; ++c) {
            // Generates a visible grid pattern: row index in upper nibble, col
            // in lower
            matrix[r][c] = c;
        }
    }

    // get the exact memory address pointer of the array block as a int64..
    // idk if I have to do this
    uint64_t src_mem = reinterpret_cast<uint64_t>(&matrix[0][0]);

    m5_work_begin(0, 0);

    amx_tile_loadd(dest_tile, src_mem, stride);

    m5_work_end(0, 0);

    return 0;
}