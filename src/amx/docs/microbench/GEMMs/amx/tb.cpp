#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string_view>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/perf_event.h>

#include "helpers.hh"
#include "gemms.hh"

// Compile:
//   g++ -std=c++23 -O3 -mamx-tile -mamx-bf16 tb.cpp -o tb
//
// Usage:
//   ./tb                     # benchmark intel_manual and bf16_gemm_6acc
//   ./tb --intel_manual      # benchmark 2x2 interleaved (Continuous Tile Layout A & B)
//   ./tb --6acc              # benchmark 3x2 interleaved (6 accumulators, Continuous Tile Layout A & B)

constexpr int WARMUP_ITERS = 20;
constexpr int MEASURE_ITERS = 100;

// A[M][K] * B[K][N] -> C[M][N].
static const int GEMM_SIZES[][3] = { // M, K, N
    {16, 512, 128000},
    {16, 128000, 512},
    {16, 512, 2048},
    {16, 2048, 512},
    {2048, 32, 2048},
    {32, 7168, 2048},
    {32, 2048, 7168},
};

// Largest buffer sizes across these shapes, also sufficient for the
// small verification cases.
constexpr int MAX_A_ELEMENTS = 16 * 128000;
constexpr int MAX_B_ELEMENTS = 512 * 128000;
constexpr int MAX_C_ELEMENTS = 2048 * 2048;

// 64-byte aligned static buffers
alignas(64) static std::bfloat16_t matrix_a[MAX_A_ELEMENTS];
alignas(64) static std::bfloat16_t packed_a[MAX_A_ELEMENTS];
alignas(64) static std::bfloat16_t matrix_b[MAX_B_ELEMENTS];
alignas(64) static std::bfloat16_t packed_b[MAX_B_ELEMENTS];
alignas(64) static float matrix_c[MAX_C_ELEMENTS];
alignas(64) static float matrix_c_ref[MAX_C_ELEMENTS];

// Offset 64-byte aligned storage by one element
alignas(64) static std::bfloat16_t unaligned_a_storage[MAX_A_ELEMENTS + 1];
alignas(64) static std::bfloat16_t unaligned_b_storage[MAX_B_ELEMENTS + 1];
alignas(64) static std::bfloat16_t unaligned_packed_b_storage[MAX_B_ELEMENTS + 1];
alignas(64) static float unaligned_c_storage[MAX_C_ELEMENTS + 1];

static std::bfloat16_t *const unaligned_a = unaligned_a_storage + 1;
static std::bfloat16_t *const unaligned_b = unaligned_b_storage + 1;
static std::bfloat16_t *const unaligned_packed_b = unaligned_packed_b_storage + 1;
static float *const unaligned_c = unaligned_c_storage + 1;

using gemm_function = void (*)(const std::bfloat16_t *, const std::bfloat16_t *, float *, int, int, int);

enum class PackingType {
    Unaligned,
    VNNI,
    Continuous,
    ContinuousB = Continuous,
    ContinuousAB,
    ContinuousA = ContinuousAB,
};

struct KernelEntry
{
    std::string_view flag;
    std::string_view name;
    gemm_function fn;
    PackingType packing;
};

const KernelEntry KERNELS[] = {
    {"--intel_manual",         "intel_manual",                 intel_manual,                 PackingType::ContinuousAB},
    {"--6acc",                 "bf16_gemm_6acc",               bf16_gemm_6acc,               PackingType::ContinuousAB},
};

bool
verify_kernel(const KernelEntry &kernel, int M = 64, int N = 128, int K = 96)
{
    const bool unaligned = kernel.packing == PackingType::Unaligned;
    std::bfloat16_t *a = unaligned ? unaligned_a : matrix_a;
    std::bfloat16_t *b = unaligned ? unaligned_b : matrix_b;
    std::bfloat16_t *b_packed = unaligned ? unaligned_packed_b : packed_b;
    float *c = unaligned ? unaligned_c : matrix_c;

    // Repeatable, varied inputs distinguish rows, columns, and K blocks.
    // Small integers are exact in BF16 and keep the FP32 reference sum exact.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> value(-8, 8);
    for (int i = 0; i < M * K; ++i) {
        a[i] = static_cast<std::bfloat16_t>(static_cast<float>(value(rng)));
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = static_cast<std::bfloat16_t>(static_cast<float>(value(rng)));
    }

    if (kernel.packing == PackingType::ContinuousAB) {
        pack_continuous_a(a, packed_a, M, K);
    }
    if (unaligned || kernel.packing == PackingType::VNNI) {
        pack_vnni_b(b, b_packed, K, N);
    } else {
        pack_continuous_b(b, b_packed, K, N);
    }
    // Unwritten output elements must fail verification, even if expected is zero.
    std::fill_n(c, M * N, std::numeric_limits<float>::quiet_NaN());

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(a[m * K + k]) * static_cast<float>(b[k * N + n]);
            }
            matrix_c_ref[m * N + n] = sum;
        }
    }

    const std::bfloat16_t *a_input =
        kernel.packing == PackingType::ContinuousAB ? packed_a : a;
    kernel.fn(a_input, b_packed, c, M, N, K);

    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(c[i]) || std::abs(c[i] - matrix_c_ref[i]) > 1e-2f) {
            int row = i / N;
            int col = i % N;
            std::fprintf(stderr, "Verification failed for %.*s (M=%d, N=%d, K=%d) at (row %d, col %d): expected %.2f, got %.2f\n",
                         static_cast<int>(kernel.name.size()), kernel.name.data(),
                         M, N, K, row, col, matrix_c_ref[i], c[i]);
            return false;
        }
    }
    return true;
}

int perf_fd = -1;

void init_perf_counter()
{
    // --- set up the counter ---
    struct perf_event_attr pe = {};
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 0; // enable perf_event

    pe.type = PERF_TYPE_HARDWARE; 
    pe.config = PERF_COUNT_HW_CPU_CYCLES; // count unhalted cpu cycles
    pe.exclude_kernel = 1; // don't count OS cycles
    pe.exclude_hv = 1;  // don't count VM cycles
    // we also have exclude_user, exclude_kernel, exclue_idele

    perf_fd = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    // [syscall, perf_event_attr, pid (0), CPU (-1 call on any cpu), group_fd (-1 this event is the leader?), any other flags ]

    // --- enable/open the counter ---
    if (perf_fd == -1) {
        perror("perf_event_open failed");
        exit(-1);
    }
}

static inline uint64_t get_cycles()
{
    uint64_t count = 0;
    _mm_lfence();
    if (read(perf_fd, &count, sizeof(count)) != sizeof(count)) {
        return 0;
    }
    _mm_lfence();
    return count;
}

// Read this to know what the benchmark is doing:
void
benchmark_kernel(const KernelEntry &kernel)
{ 
    // M=64 exercises the 6acc 48-row body plus its 16-row fallback;
    // M=32 exercises its 32-row fallback; M=16 checks the intel_manual tail.
    if (!verify_kernel(kernel) || !verify_kernel(kernel, 32, 64, 64) ||
        !verify_kernel(kernel, 16, 64, 64)) {
        std::fprintf(stderr, "Verification failed for %.*s, skipping benchmark.\n",
                     static_cast<int>(kernel.name.size()), kernel.name.data());
        return;
    }

    std::printf("\n=== Kernel: %.*s ===\n",
                static_cast<int>(kernel.name.size()), kernel.name.data());
    std::printf("%8s %8s %8s  %20s  %12s\n",
                "M", "K", "N", "Avg Cycles", "GFLOP/s");

    for (unsigned test = 0; test < sizeof(GEMM_SIZES) / sizeof(GEMM_SIZES[0]); ++test) {
        const int M = GEMM_SIZES[test][0];
        const int K = GEMM_SIZES[test][1];
        const int N = GEMM_SIZES[test][2];

        std::fill_n(matrix_a, M * K, 1.0bf16);
        std::fill_n(matrix_b, K * N, 1.5bf16);
        pack_continuous_a(matrix_a, packed_a, M, K);
        pack_continuous_b(matrix_b, packed_b, K, N);
        std::fill_n(matrix_c, M * N, std::numeric_limits<float>::quiet_NaN());

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            kernel.fn(packed_a, packed_b, matrix_c, M, N, K);
        }

        // Benchmark
        uint64_t start, end, cycles, offset;

        start = get_cycles();
        end = get_cycles();
        offset = end - start;

        const auto start_time = std::chrono::steady_clock::now();
        start = get_cycles();

        for (int i = 0; i < MEASURE_ITERS; ++i) {
            kernel.fn(packed_a, packed_b, matrix_c, M, N, K);
        }

        end = get_cycles();
        const auto end_time = std::chrono::steady_clock::now();

        cycles = end - start - offset;
        const double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        const double avg_cycles = static_cast<double>(cycles) / MEASURE_ITERS;
        const double total_flops = 2.0 * M * K * N * MEASURE_ITERS;
        const double gflops = (total_flops / elapsed_sec) / 1e9;

        // Check the actual rectangular output outside the timed region.
        const float expected = 1.5f * K;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const float actual = matrix_c[m * N + n];
                if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-2f) {
                    std::fprintf(stderr, "Verification failed for %.*s (M=%d, K=%d, N=%d) at (%d, %d)\n",
                                 static_cast<int>(kernel.name.size()), kernel.name.data(),
                                 M, K, N, m, n);
                    return;
                }
            }
        }
        std::printf("%8d %8d %8d  %20.2f  %12.2f\n",
                    M, K, N, avg_cycles, gflops);
    }
}

int
main(int argc, char **argv)
{
    if (!enable_amx()) {
        std::fputs("Error: Failed to request XTILEDATA permission from OS.\n", stderr);
        return 1;
    }

    std::string_view target_flag = (argc > 1) ? argv[1] : "--all";

    if (target_flag == "--help" || target_flag == "-h") {
        std::printf("Usage: %s [--all|--intel_manual|--6acc]\n", argv[0]);
        return 0;
    }

    init_perf_counter();

    bool matched = false;
    for (const auto &kernel : KERNELS) {
        if (target_flag == "--all" || target_flag == kernel.flag ||
            (kernel.flag == "--intel_manual" && target_flag == "--med") ||
            (kernel.flag == "--6acc" && target_flag == "--cont")) {
            benchmark_kernel(kernel);
            matched = true;
        }
    }

    if (!matched) {
        std::fprintf(stderr, "Unknown option: %s\n", argv[1]);
        std::fprintf(stderr, "Usage: %s [--all|--intel_manual|--6acc]\n", argv[0]);
        close(perf_fd);
        return 1;
    }

    close(perf_fd);
    return 0;
}
