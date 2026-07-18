#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <x86intrin.h>

#define ITERATIONS 100000
#define STRIDE 64

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// Align memory to 64 bytes for AMX
alignas(64) int8_t src_mem[16 * STRIDE];
alignas(64) int8_t dst_mem[16 * STRIDE];

// 64-byte TILECONFIG structure
typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg_t;

static bool
set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    }
    return true;
}

void
init_amx()
{
    tilecfg_t cfg = {0};
    cfg.palette_id = 1;
    cfg.start_row = 0;

    // Configure TMM0 - TMM7 to be 16 rows by 64 bytes
    for (int i = 0; i < 8; i++) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }

    _tile_loadconfig(&cfg);
}

int
main()
{
    // Request permission from Linux kernel to use AMX
    if (!set_tiledata_use()) {
        fprintf(stderr, "Failed to enable AMX (XTILEDATA) state.\n");
        exit(-1);
    }

    init_amx();

    uint64_t start, end;
    double cycles;

    // Initialize memory
    for (int i = 0; i < 16 * STRIDE; i++) {
        src_mem[i] = i % 127;
    }

    // ==========================================
    // TEST 1: Independent Loads (TMM0 -> TMM3)
    //
    // PURPOSE:
    // This test verifies the pipeline capacity and out-of-order execution 
    // when there are no dependencies between AMX instructions.
    //
    // DETAILS:
    // It loads data into four different tile registers (TMM0 to TMM3) 
    // in each iteration. Since these registers are completely independent, 
    // an Out-of-Order CPU can issue and overlap these memory loads, 
    // bottlenecked only by the memory pipeline's throughput (~8 cycles/load).
    // Expected avg cycles per iteration: ~32 cycles (4 * 8 cycles).
    // ==========================================
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(1, src_mem, STRIDE);
        _tile_loadd(2, src_mem, STRIDE);
        _tile_loadd(3, src_mem, STRIDE);
        asm volatile("" ::: "memory"); // Prevent compiler from optimizing out
    }
    end = _rdtsc();
    cycles = (double)(end - start) / ITERATIONS;
    printf("Independent TILELOADD (TMM0, TMM1, TMM2, TMM3) avg cycles: %.2f\n",
           cycles);

    // ==========================================
    // TEST 2: Dependent Loads (WAW on TMM0)
    //
    // PURPOSE:
    // This test checks for the existence of Register Renaming for AMX tiles 
    // by introducing a Write-After-Write (WAW) false dependency.
    //
    // DETAILS:
    // Four loads write sequentially to the SAME register TMM0. 
    // - If register renaming is present (like in vector/GP registers), the 
    //   hazard is eliminated and these loads will execute in parallel (avg ~32 cycles).
    // - If renaming is NOT present (real AMX hardware), each write to TMM0 must 
    //   physically serialize and wait for the prior write to finish.
    // Expected avg cycles per iteration: ~180 cycles (4 * 45 cycles latency).
    // ==========================================
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    cycles = (double)(end - start) / ITERATIONS;
    printf("Dependent TILELOADD (WAW on TMM0) avg cycles: %.2f\n", cycles);

    // ==========================================
    // TEST 3: Independent Compute & Load (No Hazard)
    //
    // PURPOSE:
    // This test evaluates the ability of the CPU to overlap independent compute 
    // (TMUL) and memory loads in parallel.
    //
    // DETAILS:
    // A matrix multiplication (TDPBUSD) writes to TMM2 using TMM0 and TMM1 as 
    // inputs. Simultaneously, an independent TILELOADD writes to TMM3.
    // Since there are no dependencies between TMM0/TMM1/TMM2 and TMM3, the load 
    // executes in parallel with the compute.
    // Expected avg cycles per iteration: ~52 cycles (max of compute/load latency).
    // ==========================================
    _tile_loadd(0, src_mem, STRIDE);
    _tile_loadd(1, src_mem, STRIDE);
    _tile_loadd(2, src_mem, STRIDE);
    _tile_zero(3);

    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(3, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    cycles = (double)(end - start) / ITERATIONS;
    printf("Independent Compute + Load (TMM3) avg cycles: %.2f\n", cycles);

    // ==========================================
    // TEST 4: Dependent Compute & Load (WAR Hazard on TMM0)
    //
    // PURPOSE:
    // This test verifies execution behavior under a Write-After-Read (WAR) hazard.
    //
    // DETAILS:
    // The TDPBUSD compute instruction reads TMM0 and TMM1. The subsequent 
    // TILELOADD writes to TMM0.
    // Without register renaming, the write to TMM0 cannot map to a new physical 
    // register. The load must stall until the compute instruction finishes 
    // reading TMM0.
    // Expected avg cycles per iteration: ~97 cycles (compute + load latency).
    // ==========================================
    _tile_zero(0);
    _tile_loadd(1, src_mem, STRIDE);
    _tile_loadd(2, src_mem, STRIDE);

    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    cycles = (double)(end - start) / ITERATIONS;
    printf("Dependent Compute + Load (WAR on TMM0) avg cycles: %.2f\n",
           cycles);

    // ==========================================
    // SWEEP TEST: 1 to 16 Loads
    //
    // PURPOSE:
    // Sweep the number of loads from 1 to 16 to find the exact point where
    // the physical register file (PRF) rename pool is exhausted.
    // ==========================================
#define RUN_LOAD_SWEEP(n, ...) \
    start = _rdtsc(); \
    for (int i = 0; i < ITERATIONS; i++) { \
        __VA_ARGS__ \
        asm volatile("" ::: "memory"); \
    } \
    end = _rdtsc(); \
    cycles = (double)(end - start) / ITERATIONS; \
    printf("Loads Sweep - %2d Loads avg cycles: %6.2f (per load: %5.2f)\n", n, cycles, cycles / (n));

    RUN_LOAD_SWEEP(1, _tile_loadd(0, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(2, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(3, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(4, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(5, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(6, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(7, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(8, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(9, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(10, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(11, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(12, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(13, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(14, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(15, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE);)
    RUN_LOAD_SWEEP(16, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE);)
#undef RUN_LOAD_SWEEP

    _tile_release();
    return 0;
}

// gcc -O3 -march=native bench.c -o bench