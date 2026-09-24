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

    // Configure TMM0 - TMM7 to be 16 rows by 64 bytes (1 KB each)
    for (int i = 0; i < 8; i++) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }

    _tile_loadconfig(&cfg);
}

int main()
{
    // Request permission from Linux kernel to use AMX
    if (!set_tiledata_use()) {
        fprintf(stderr, "Failed to enable AMX (XTILEDATA) state.\n");
        exit(-1);
    }

    init_amx();

    // Initialize source/destination memory
    for (int i = 0; i < 16 * STRIDE; i++) {
        src_mem[i] = (i % 127) + 1;
        dst_mem[i] = ((i + 13) % 127) + 1;
    }

    uint64_t start, end;
    double cycles;

    // Populate tile registers with initial non-zero data
    _tile_loadd(0, src_mem, STRIDE);
    _tile_loadd(1, src_mem, STRIDE);
    _tile_loadd(2, src_mem, STRIDE);
    _tile_loadd(3, src_mem, STRIDE);
    _tile_loadd(4, src_mem, STRIDE);
    _tile_loadd(5, src_mem, STRIDE);
    _tile_loadd(6, src_mem, STRIDE);
    _tile_loadd(7, src_mem, STRIDE);

    printf("=====================================================================\n");
    printf("Intel AMX Out-of-Order Capabilities, Renaming, & Hazard Tests\n");
    printf("=====================================================================\n\n");

    // =====================================================================
    // SECTION 1: Can Compute Executions Happen in Parallel?
    // =====================================================================
    printf("--- SECTION 1: Compute Execution Port Concurrency ---\n");

    // 1A: Single Matrix Multiply
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_1a = (double)(end - start) / ITERATIONS;
    printf("1A: 1 Compute (dpbusd) avg cycles: %.2f\n", cycles_1a);

    // 1B: Two Independent Matrix Multiplies
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_dpbusd(5, 3, 4);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_1b = (double)(end - start) / ITERATIONS;
    printf("1B: 2 Independent Computes (dpbusd) avg cycles: %.2f\n", cycles_1b);
    printf("  -> Execution Unit Port Scaling: %.2f (1.00 = Fully Parallel, 2.00 = Fully Serialized)\n\n", 
           cycles_1b / cycles_1a);


    // =====================================================================
    // SECTION 2: Can Loads & Stores Happen in Parallel?
    // =====================================================================
    printf("--- SECTION 2: Memory Pipeline Concurrency ---\n");

    // 2A: Load Alone
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_2a = (double)(end - start) / ITERATIONS;
    printf("2A: 1 Tile Load (TILELOADD) alone avg cycles: %.2f\n", cycles_2a);

    // 2B: Store Alone
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_stored(1, dst_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_2b = (double)(end - start) / ITERATIONS;
    printf("2B: 1 Tile Store (TILESTORED) alone avg cycles: %.2f\n", cycles_2b);

    // 2C: Store + Independent Load
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_stored(1, dst_mem, STRIDE);
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_2c = (double)(end - start) / ITERATIONS;
    printf("2C: 1 Store + 1 Independent Load avg cycles: %.2f\n", cycles_2c);
    
    double max_single_mem = (cycles_2a > cycles_2b) ? cycles_2a : cycles_2b;
    double mem_parallelism = (cycles_2a + cycles_2b - cycles_2c) / (cycles_2a < cycles_2b ? cycles_2a : cycles_2b);
    if (mem_parallelism < 0) mem_parallelism = 0;
    if (mem_parallelism > 1) mem_parallelism = 1;
    printf("  -> Load & Store Parallelism Overlap: %.2f%% (100%% = Fully Parallel, 0%% = Fully Serialized)\n\n",
           mem_parallelism * 100.0);


    // =====================================================================
    // SECTION 3: Resolving of Register Hazards (WAW, WAR, RAW)
    // =====================================================================
    printf("--- SECTION 3: Hazard Resolution & Register Renaming ---\n");

    // 3A: WAW Hazard on Load
    // Baseline: 2 independent loads (TMM0, TMM1)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(1, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3a_indep = (double)(end - start) / ITERATIONS;
    
    // Test: 2 dependent loads to same register (TMM0, TMM0)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3a_dep = (double)(end - start) / ITERATIONS;
    printf("3A: WAW Hazard on Loads (Load-after-Load to same reg):\n");
    printf("  - Independent: %.2f cycles | Dependent: %.2f cycles\n", cycles_3a_indep, cycles_3a_dep);
    printf("  - WAW Renaming Status: %s\n\n", (cycles_3a_dep < cycles_3a_indep * 1.1) ? "RENAME SUCCESSFUL (Resolved)" : "STALLED/SERIALIZED (No Renaming)");

    // 3B: WAR Hazard between Compute and Load
    // Baseline: Compute (reads TMM0), Independent Load (writes TMM3)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(3, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3b_indep = (double)(end - start) / ITERATIONS;

    // Test: Compute (reads TMM0), Dependent Load (writes TMM0 - WAR Hazard)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(0, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3b_dep = (double)(end - start) / ITERATIONS;
    printf("3B: WAR Hazard on Compute & Load (Write TMM0 after Compute Reads TMM0):\n");
    printf("  - Independent: %.2f cycles | Dependent (WAR): %.2f cycles\n", cycles_3b_indep, cycles_3b_dep);
    printf("  - WAR Renaming Status: %s\n\n", (cycles_3b_dep < cycles_3b_indep * 1.1) ? "RENAME SUCCESSFUL (Resolved)" : "STALLED/SERIALIZED (No Renaming)");

    // 3C: WAW Hazard between Compute and Load
    // Baseline: Compute (writes TMM2), Independent Load (writes TMM3)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(3, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3c_indep = (double)(end - start) / ITERATIONS;

    // Test: Compute (writes TMM2), Dependent Load (writes TMM2 - WAW Hazard)
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_dpbusd(2, 0, 1);
        _tile_loadd(2, src_mem, STRIDE);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3c_dep = (double)(end - start) / ITERATIONS;
    printf("3C: WAW Hazard on Compute & Load (Write TMM2 after Compute Writes TMM2):\n");
    printf("  - Independent: %.2f cycles | Dependent (WAW): %.2f cycles\n", cycles_3c_indep, cycles_3c_dep);
    printf("  - WAW Renaming Status: %s\n\n", (cycles_3c_dep < cycles_3c_indep * 1.1) ? "RENAME SUCCESSFUL (Resolved)" : "STALLED/SERIALIZED (No Renaming)");

    // 3D: True RAW Dependency (Baseline for Serialization)
    // Test: Load (writes TMM0), Compute (reads TMM0 - RAW Hazard)
    // This is a true dependency and cannot be resolved by renaming.
    start = _rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        _tile_loadd(0, src_mem, STRIDE);
        _tile_dpbusd(2, 0, 1);
        asm volatile("" ::: "memory");
    }
    end = _rdtsc();
    double cycles_3d_raw = (double)(end - start) / ITERATIONS;
    printf("3D: True RAW Dependency (Compute reads TMM0 immediately after Load writes TMM0):\n");
    printf("  - Serialized Latency: %.2f cycles\n\n", cycles_3d_raw);


    // =====================================================================
    // SECTION 4: Register Renaming Limits (Sweep 1 to 16 Loads)
    // =====================================================================
    printf("--- SECTION 4: Register Renaming Limits (Sweep 1-16) ---\n");

#define SWEEP_LOADS(n, ...) \
    start = _rdtsc(); \
    for (int i = 0; i < ITERATIONS; i++) { \
        __VA_ARGS__ \
        asm volatile("" ::: "memory"); \
    } \
    end = _rdtsc(); \
    cycles = (double)(end - start) / ITERATIONS; \
    printf("  %2d Loads: %6.2f cycles (per load: %5.2f)\n", n, cycles, cycles / (n));

    SWEEP_LOADS(1, _tile_loadd(0, src_mem, STRIDE);)
    SWEEP_LOADS(2, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE);)
    SWEEP_LOADS(3, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE);)
    SWEEP_LOADS(4, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE);)
    SWEEP_LOADS(5, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE);)
    SWEEP_LOADS(6, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE);)
    SWEEP_LOADS(7, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE);)
    SWEEP_LOADS(8, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE);)
    SWEEP_LOADS(9, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE);)
    SWEEP_LOADS(10, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE);)
    SWEEP_LOADS(11, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE);)
    SWEEP_LOADS(12, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE);)
    SWEEP_LOADS(13, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE);)
    SWEEP_LOADS(14, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE);)
    SWEEP_LOADS(15, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE);)
    SWEEP_LOADS(16, _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE); _tile_loadd(0, src_mem, STRIDE); _tile_loadd(1, src_mem, STRIDE); _tile_loadd(2, src_mem, STRIDE); _tile_loadd(3, src_mem, STRIDE); _tile_loadd(4, src_mem, STRIDE); _tile_loadd(5, src_mem, STRIDE); _tile_loadd(6, src_mem, STRIDE); _tile_loadd(7, src_mem, STRIDE);)
#undef SWEEP_LOADS

    _tile_release();
    return 0;
}
