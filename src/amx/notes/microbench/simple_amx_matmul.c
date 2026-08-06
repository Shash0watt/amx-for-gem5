#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

// compile using: gcc -O3 -mamx-tile -mamx-bf16 src/amx/notes/microbench/simple_amx_matmul.c -o simple_amx_matmul

// 64-byte AMX Tile Configuration Register structure
typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

// Request OS permission to use Intel AMX tile hardware
static bool set_tiledata_use(void)
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        return false;
    }
    return true;
}

int main(void)
{
    // 1. Request OS permission for AMX tile data
    if (!set_tiledata_use()) {
        fprintf(stderr, "Error: AMX tile permission (XFEATURE_XTILEDATA) failed.\n");
        return -1;
    }

    // 2. Input matrix setup:
    // Dest tile (Tile 0): One FP32 element initialized to 1.0F
    alignas(64) float dest_tile[1][1] = {{1.0F}};

    // Tile A (Tile 1): One pair of BF16 inputs (2.0 = 0x4000, 3.0 = 0x4040)
    alignas(64) uint16_t tile_a[1][2] = {{0x4000, 0x4040}};

    // Tile B (Tile 2): One pair of BF16 inputs (4.0 = 0x4080, 5.0 = 0x40a0)
    alignas(64) uint16_t tile_b[1][2] = {{0x4080, 0x40a0}};

    // Prepare result storage
    alignas(64) float result[1][1] = {{0.0F}};

    // 3. Tile configuration setup
    alignas(64) __tilecfg config = {0};
    config.palette_id = 1;
    config.start_row = 0;

    for (int tile = 0; tile < 3; ++tile) {
        config.colsb[tile] = sizeof(uint32_t); // 4 bytes per row
        config.rows[tile] = 1;                 // 1 row
    }

    // 4. Warmup loop (Wakes up AMX hardware TMUL, handles kernel XSAVE allocation, warms caches)
    const int warmup_iterations = 1000;
    for (int i = 0; i < warmup_iterations; i++) {
        _tile_loadconfig(&config);
        _tile_loadd(0, dest_tile, sizeof(dest_tile[0]));
        _tile_loadd(1, tile_a, sizeof(tile_a[0]));
        _tile_loadd(2, tile_b, sizeof(tile_b[0]));
        _tile_dpbf16ps(0, 1, 2);
        _tile_stored(0, result, sizeof(result[0]));
    }

    // 5. Timed warm execution measurement
    unsigned int dummy;
    uint64_t start_cycles = __rdtscp(&dummy);

    // Load tile configuration into hardware
    _tile_loadconfig(&config);

    // Load input tiles into AMX registers TMM0, TMM1, TMM2
    _tile_loadd(0, dest_tile, sizeof(dest_tile[0]));
    _tile_loadd(1, tile_a, sizeof(tile_a[0]));
    _tile_loadd(2, tile_b, sizeof(tile_b[0]));

    // Perform BF16 dot-product multiply-accumulate:
    // TMM0 = TMM0 + (TMM1 * TMM2)
    // Expected result: 1.0 + (2.0 * 4.0) + (3.0 * 5.0) = 24.0
    _tile_dpbf16ps(0, 1, 2);

    // Store accumulated result from TMM0 back to memory
    _tile_stored(0, result, sizeof(result[0]));

    uint64_t end_cycles = __rdtscp(&dummy);
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // 6. Release tile configuration
    _tile_release();

    // 7. Verify and print result & cycle count
    printf("AMX TDPBF16PS Computation Result: %.1f (Expected: 24.0)\n", result[0][0]);
    printf("Warm Execution CPU Cycles (loadconfig to stored): %lu cycles\n", (unsigned long)elapsed_cycles);

    return 0;
}
