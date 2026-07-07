#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib> // for atoi

#include <gem5/m5ops.h> // for gem5 magic ops

int
main(int argc, char *argv[])
{
    uint64_t dest_tile = 0;
    size_t stride = 64;

    // TESTING VARIABLES:
    // change this to change the offset
    int offset = 60;

    // allocate a larger raw buffer that is aligned to 64 bytes.
    // this gives us a predictable baseline.
    alignas(64) int8_t raw_buffer[16 * 64 + 64] = {0};

    // cast it to a pointer to an array of 64 elements.
    // passing 0 keeps it aligned. passing 60 forces it across 2 cache lines.
    auto matrix = reinterpret_cast<int8_t (*)[64]>(&raw_buffer[offset]);

    // fill the matrix
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 64; ++c) {
            matrix[r][c] = c;
        }
    }

    m5_work_begin(0, 0);

    amx_tile_loadd(dest_tile, &matrix[0][0], stride);

    m5_work_end(0, 0);

    return 0;
}