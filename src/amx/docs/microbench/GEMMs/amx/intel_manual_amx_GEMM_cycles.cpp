// Cycle count of the intel_manual 2x2-tile BF16 GEMM (intel_manual_amx_GEMM.cpp)
//   g++ -O2 -mamx-tile -mamx-bf16 cycles.cpp -o cycles

#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// The 64-byte memory layout consumed by LDTILECFG.
struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

constexpr int TM = 16;
constexpr int TN = 16;
constexpr int TK = 32;

// A is 16 x 2048, B is 2048 x 512, and C is 16 x 512.
constexpr int M = 16;
constexpr int N = 512;
constexpr int K = 2048;

// Read the core cycle counter (same as microbench/intrinsic/load_trgh.c).
static inline uint64_t
getCycles(int perf_fd)
{
    uint64_t count = 0;
    _mm_lfence();
    read(perf_fd, &count, sizeof(count));
    _mm_lfence();
    return count;
}

// Wall-clock time in nanoseconds (vDSO call, no syscall).
static inline uint64_t
getNanos()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// The intel_manual kernel: C = A * B with a 2x2 block of accumulator tiles.
void
gemm(const __bf16 *packed_a, const __bf16 *packed_b, float *C)
{
    constexpr int stride_a = 32 * sizeof(__bf16);
    constexpr int stride_b = 32 * sizeof(__bf16);
    constexpr int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * TM) {
        const bool has_bottom = m + TM < M;
        for (int n = 0; n < N; n += 2 * TN) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            const __bf16 *a_top = packed_a + m * K;
            const __bf16 *a_bot = packed_a + (m + TM) * K;
            const __bf16 *b_ptr = packed_b + n * K;

            for (int k = 0; k < K; k += TK) {
                const __bf16 *b_left = b_ptr;
                const __bf16 *b_right = b_ptr + 512; // 512 elements = 1024 bytes

                _tile_loadd(4, a_top, stride_a);
                _tile_loadd(6, b_left, stride_b);
                _tile_dpbf16ps(0, 4, 6); // C00 += A_top * B_left

                if (has_bottom) {
                    _tile_loadd(5, a_bot, stride_a);
                    _tile_dpbf16ps(2, 5, 6); // C10 += A_bot * B_left
                }

                _tile_loadd(7, b_right, stride_b);
                _tile_dpbf16ps(1, 4, 7); // C01 += A_top * B_right
                if (has_bottom) {
                    _tile_dpbf16ps(3, 5, 7); // C11 += A_bot * B_right
                }

                a_top += TM * TK; // 512 elements = 1024 bytes
                if (has_bottom) {
                    a_bot += TM * TK;
                }
                b_ptr += 1024; // 1024 elements = 2048 bytes
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + TN, stride_c);
            if (has_bottom) {
                _tile_stored(2, C + (m + TM) * N + n, stride_c);
                _tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
            }
        }
    }
}

int
main()
{
    // Linux only lets a process use AMX tile data after asking for permission.
    constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr int XFEATURE_XTILEDATA = 18;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        std::fprintf(stderr, "Failed to get AMX permission from the OS\n");
        return 1;
    }

   // setup cycles counter
    perf_event_attr pe = {};
    pe.size = sizeof(pe);
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    const int perf_fd = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (perf_fd == -1) {
        std::perror("perf_event_open failed");
        return 1;
    }

    // Inputs cycle through zeros, integers and decimals.
    const float values[] = {0.0F, 1.0F, -2.0F, 3.0F, 0.5F, -1.25F, 0.1F, 2.75F};
    static __bf16 A[M][K], B[K][N];
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            A[m][k] = static_cast<__bf16>(values[(m + k) % 8]);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B[k][n] = static_cast<__bf16>(values[(3 * k + n) % 8]);
        }
    }

    // Pack A into Continuous Tile Layout
    alignas(64) static __bf16 packed_a[M * K];
    int out = 0;
    for (int m = 0; m < M; m += TM) {
        for (int k = 0; k < K; k += TK) {
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TK; ++c) {
                    packed_a[out++] = A[m + r][k + c];
                }
            }
        }
    }

    // Pack B into Continuous Tile Layout (as needed by intel amx)
    alignas(64) static __bf16 packed_b[K * N];
    out = 0;
    for (int n = 0; n < N; n += 2 * TN) {
        for (int k = 0; k < K; k += TK) {
            for (int half = 0; half < 2; ++half) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        packed_b[out++] = B[k + 2 * r][n + half * TN + c];
                        packed_b[out++] = B[k + 2 * r + 1][n + half * TN + c];
                    }
                }
            }
        }
    }

    alignas(64) static float C[M * N];

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.columnBytes[tile] = 64; // 32 BF16 values
    }

    _tile_loadconfig(&config);

    // Warm-up run (not timed): pays for first AMX use (the OS allocating tile
    // state, AMX power-up) and brings A, B and C into the caches.
    gemm(packed_a, packed_b, C);

    // Cost of reading the counter itself, subtracted from the result below.
    uint64_t start = getCycles(perf_fd);
    uint64_t end = getCycles(perf_fd);
    const uint64_t offset = end - start;

    // The wall-clock reads sit inside the cycle reads so the time spent in the
    // read() syscalls (not counted as user cycles) is kept out of the timing.
    start = getCycles(perf_fd);
    const uint64_t t0 = getNanos();
    gemm(packed_a, packed_b, C);
    const uint64_t t1 = getNanos();
    end = getCycles(perf_fd);

    _tile_release();
    close(perf_fd);

    const uint64_t cycles = end - start - offset;
    const uint64_t nanos = t1 - t0;

    // Average core frequency over the timed run: cycles per nanosecond = GHz.
    std::printf("Cycles Taken: %lu\n", cycles);
    std::printf("Time Taken: %lu ns\n", nanos);
    std::printf("Average Frequency: %.3f GHz\n",
                static_cast<double>(cycles) / static_cast<double>(nanos));

    return 0;
}
