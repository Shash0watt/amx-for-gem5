#include <cstdint>
#include <gem5/m5ops.h>
#include <iostream>

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

    std::cout << "multiple tile loads test" << std::endl;

    m5_work_begin(0, 0);

    // load array1 into tile 0
    amx_tile_loadd(0, array1, stride);

    // load array2 into tile 0
    size_t stride2 = 16 * sizeof(int32_t);
    amx_tile_loadd(0, array2, stride);

    m5_work_end(0, 0);

    std::cout << "multiple tile loads test" << std::endl;

    return 0;
}
