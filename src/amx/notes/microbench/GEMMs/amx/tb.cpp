#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <immintrin.h>

#include "helpers.hh"
#include "gemms.hh"

// Compile:
//   g++ -std=c++23 -O3 -mamx-tile -mamx-bf16 tb.cpp -o tb
//
// Usage:
//   ./tb                     # benchmark all kernels
//   ./tb --none              # benchmark 1x1 basic (unaligned loads)
//   ./tb --aligned           # benchmark 1x1 basic (64-byte aligned loads)
//   ./tb --aligned_cont_b    # benchmark 1x1 basic (Continuous Tile Layout B, strided A)
//   ./tb --aligned_cont_ab   # benchmark 1x1 basic (Continuous Tile Layout A & B)
//   ./tb --intel_manual      # benchmark 2x2 interleaved (Continuous Tile Layout A & B)
//   ./tb --6acc              # benchmark 3x2 interleaved (6 accumulators, Continuous Tile Layout A & B)

constexpr int WARMUP_ITERS = 20;
constexpr int MEASURE_ITERS = 100;
constexpr int MATRIX_SIZES[] = {32, 64, 128, 256, 512, 1024};
constexpr int MAX_SIZE = 1024;

// 64-byte aligned static buffers
alignas(64) static std::bfloat16_t matrix_a[MAX_SIZE * MAX_SIZE];
alignas(64) static std::bfloat16_t packed_a[MAX_SIZE * MAX_SIZE];
alignas(64) static std::bfloat16_t matrix_b[MAX_SIZE * MAX_SIZE];
alignas(64) static std::bfloat16_t packed_b[MAX_SIZE * MAX_SIZE];
alignas(64) static float matrix_c[MAX_SIZE * MAX_SIZE];
alignas(64) static float matrix_c_ref[MAX_SIZE * MAX_SIZE];

// Unaligned static buffers for unaligned baseline
static std::bfloat16_t unaligned_a[MAX_SIZE * MAX_SIZE];
static std::bfloat16_t unaligned_b[MAX_SIZE * MAX_SIZE];
static std::bfloat16_t unaligned_packed_b[MAX_SIZE * MAX_SIZE];
static float unaligned_c[MAX_SIZE * MAX_SIZE];
static float unaligned_c_ref[MAX_SIZE * MAX_SIZE];

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
    {"--none",                 "bf16_gemm_none",               bf16_gemm_none,               PackingType::Unaligned},
    {"--aligned",              "bf16_gemm_aligned",            bf16_gemm_aligned,            PackingType::VNNI},
    {"--aligned_cont_b",       "bf16_gemm_aligned_cont_b",     bf16_gemm_aligned_cont_b,     PackingType::ContinuousB},
    {"--aligned_cont_ab",      "bf16_gemm_aligned_cont_ab",    bf16_gemm_aligned_cont_ab,    PackingType::ContinuousAB},
    {"--intel_manual",         "intel_manual",                 intel_manual,                 PackingType::ContinuousAB},
    {"--6acc",                 "6acc",                         bf16_gemm_6acc,               PackingType::ContinuousAB},
};

bool
verify_kernel(const KernelEntry &kernel, int M = 64, int N = 128, int K = 96)
{
    if (kernel.packing == PackingType::Unaligned) {
        for (int i = 0; i < M * K; ++i) {
            unaligned_a[i] = static_cast<std::bfloat16_t>(static_cast<float>((i % 4) + 1));
        }
        for (int i = 0; i < K * N; ++i) {
            unaligned_b[i] = static_cast<std::bfloat16_t>(static_cast<float>(4 - (i % 4)));
        }

        pack_vnni_b(unaligned_b, unaligned_packed_b, K, N);
        std::fill_n(unaligned_c, M * N, 0.0f);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += static_cast<float>(unaligned_a[m * K + k]) * static_cast<float>(unaligned_b[k * N + n]);
                }
                unaligned_c_ref[m * N + n] = sum;
            }
        }

        kernel.fn(unaligned_a, unaligned_packed_b, unaligned_c, M, N, K);

        for (int i = 0; i < M * N; ++i) {
            if (std::abs(unaligned_c[i] - unaligned_c_ref[i]) > 1e-2f) {
                int row = i / N;
                int col = i % N;
                std::fprintf(stderr, "Verification failed for %.*s at (row %d, col %d): expected %.2f, got %.2f\n",
                             static_cast<int>(kernel.name.size()), kernel.name.data(),
                             row, col, unaligned_c_ref[i], unaligned_c[i]);
                return false;
            }
        }
        return true;
    }

    for (int i = 0; i < M * K; ++i) {
        matrix_a[i] = static_cast<std::bfloat16_t>(static_cast<float>((i % 4) + 1));
    }
    for (int i = 0; i < K * N; ++i) {
        matrix_b[i] = static_cast<std::bfloat16_t>(static_cast<float>(4 - (i % 4)));
    }

    if (kernel.packing == PackingType::ContinuousAB) {
        pack_continuous_a(matrix_a, packed_a, M, K);
        pack_continuous_b(matrix_b, packed_b, K, N);
    } else if (kernel.packing == PackingType::VNNI) {
        pack_vnni_b(matrix_b, packed_b, K, N);
    } else {
        pack_continuous_b(matrix_b, packed_b, K, N);
    }
    std::fill_n(matrix_c, M * N, 0.0f);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(matrix_a[m * K + k]) * static_cast<float>(matrix_b[k * N + n]);
            }
            matrix_c_ref[m * N + n] = sum;
        }
    }

    if (kernel.packing == PackingType::ContinuousAB) {
        kernel.fn(packed_a, packed_b, matrix_c, M, N, K);
    } else {
        kernel.fn(matrix_a, packed_b, matrix_c, M, N, K);
    }

    for (int i = 0; i < M * N; ++i) {
        if (std::abs(matrix_c[i] - matrix_c_ref[i]) > 1e-2f) {
            int row = i / N;
            int col = i % N;
            std::fprintf(stderr, "Verification failed for %.*s at (row %d, col %d): expected %.2f, got %.2f\n",
                         static_cast<int>(kernel.name.size()), kernel.name.data(),
                         row, col, matrix_c_ref[i], matrix_c[i]);
            return false;
        }
    }
    return true;
}

// Read this to know what the benchmark is doing:
void
benchmark_kernel(const KernelEntry &kernel)
{ 
    if (!verify_kernel(kernel)) {
        std::fprintf(stderr, "Verification failed for %.*s, skipping benchmark.\n",
                     static_cast<int>(kernel.name.size()), kernel.name.data());
        return;
    }

    std::printf("\n=== Kernel: %.*s ===\n",
                static_cast<int>(kernel.name.size()), kernel.name.data());
    std::printf("%8s  %20s  %12s\n", "Size", "Avg TSC Cycles", "GFLOP/s");
    
    for (int size : MATRIX_SIZES) {
        std::bfloat16_t *a_ptr;
        std::bfloat16_t *b_ptr;
        float *c_ptr;

        if (kernel.packing == PackingType::Unaligned) {
            std::fill_n(unaligned_a, size * size, 1.0bf16);
            std::fill_n(unaligned_b, size * size, 1.5bf16);
            pack_vnni_b(unaligned_b, unaligned_packed_b, size, size);
            std::fill_n(unaligned_c, size * size, 0.0f);
            a_ptr = unaligned_a;
            b_ptr = unaligned_packed_b;
            c_ptr = unaligned_c;
        } else if (kernel.packing == PackingType::ContinuousAB) {
            std::fill_n(matrix_a, size * size, 1.0bf16);
            std::fill_n(matrix_b, size * size, 1.5bf16);
            pack_continuous_a(matrix_a, packed_a, size, size);
            pack_continuous_b(matrix_b, packed_b, size, size);
            std::fill_n(matrix_c, size * size, 0.0f);
            a_ptr = packed_a;
            b_ptr = packed_b;
            c_ptr = matrix_c;
        } else {
            std::fill_n(matrix_a, size * size, 1.0bf16);
            std::fill_n(matrix_b, size * size, 1.5bf16);
            if (kernel.packing == PackingType::VNNI) {
                pack_vnni_b(matrix_b, packed_b, size, size);
            } else {
                pack_continuous_b(matrix_b, packed_b, size, size);
            }
            std::fill_n(matrix_c, size * size, 0.0f);
            a_ptr = matrix_a;
            b_ptr = packed_b;
            c_ptr = matrix_c;
        }

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            kernel.fn(a_ptr, b_ptr, c_ptr, size, size, size);
        }

        // Benchmark
        const auto start_time = std::chrono::steady_clock::now();
        _mm_lfence();
        const std::uint64_t start_cycles = __rdtsc();
        _mm_lfence();

        for (int i = 0; i < MEASURE_ITERS; ++i) {
            kernel.fn(a_ptr, b_ptr, c_ptr, size, size, size);
        }

        unsigned int aux;
        const std::uint64_t end_cycles = __rdtscp(&aux);
        _mm_lfence();
        const auto end_time = std::chrono::steady_clock::now();

        const double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        const double avg_cycles = static_cast<double>(end_cycles - start_cycles) / MEASURE_ITERS;
        const double total_flops = 2.0 * size * size * size * MEASURE_ITERS;
        const double gflops = (total_flops / elapsed_sec) / 1e9;

        std::printf("%4d^3    %20.2f  %12.2f\n",
                    size, avg_cycles, gflops);
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
        std::printf("Usage: %s [--all|--none|--aligned|--aligned_cont_b|--aligned_cont_ab|--intel_manual|--6acc]\n", argv[0]);
        return 0;
    }

    bool matched = false;
    for (const auto &kernel : KERNELS) {
        if (target_flag == "--all" || target_flag == kernel.flag ||
            (kernel.flag == "--aligned_cont_b" && target_flag == "--aligned_cont") ||
            (kernel.flag == "--intel_manual" && target_flag == "--med") ||
            (kernel.flag == "--6acc" && target_flag == "--cont")) {
            benchmark_kernel(kernel);
            matched = true;
        }
    }

    if (!matched) {
        std::fprintf(stderr, "Unknown option: %s\n", argv[1]);
        std::fprintf(stderr, "Usage: %s [--all|--none|--aligned|--aligned_cont_b|--aligned_cont_ab|--intel_manual|--6acc]\n", argv[0]);
        return 1;
    }

    return 0;
}
