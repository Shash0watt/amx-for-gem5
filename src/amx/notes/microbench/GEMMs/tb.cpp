#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>
#include <immintrin.h>

#include "helpers.hh"
#include "gemms.hh"

// Compile:
//   g++ -std=c++23 -O3 -mamx-tile -mamx-bf16 tb.cpp -o tb
//
// Usage:
//   ./tb          # benchmark all kernels
//   ./tb --none   # benchmark 1x1 basic
//   ./tb --low    # benchmark 2x2 sequential
//   ./tb --med    # benchmark 2x2 interleaved
//   ./tb --opt    # benchmark 2x2 cache-blocked + prefetch

constexpr int WARMUP_ITERS = 20;
constexpr int MEASURE_ITERS = 100;
constexpr int MATRIX_SIZES[] = {128, 256, 512, 1024};

using gemm_function = void (*)(const std::bfloat16_t *, const std::bfloat16_t *, float *, int, int, int);

struct KernelEntry
{
    std::string_view flag;
    std::string_view name;
    gemm_function fn;
};

const KernelEntry KERNELS[] = {
    {"--none", "bf16_gemm_none (1x1 basic)",         bf16_gemm_none},
    {"--low",  "bf16_gemm_low  (2x2 sequential)",    bf16_gemm_low},
    {"--med",  "bf16_gemm_med  (2x2 interleaved)",   bf16_gemm_med},
    {"--opt",  "bf16_gemm_opt  (2x2 blocked+pref)",  bf16_gemm_opt},
};


bool
verify_kernel(gemm_function gemm, int size = 64)
{
    // First initlaiize the data
    const auto A = std::vector<std::bfloat16_t>(size * size, 1.0bf16);
    const auto raw_B = std::vector<std::bfloat16_t>(size * size, 2.0bf16);
    const auto B = pack_vnni_b(raw_B.data(), size, size);
    auto C = std::vector<float>(size * size, 0.0f);

    // Execute the gemm kernel
    gemm(A.data(), B.data(), C.data(), size, size, size);
    const float correct_value = static_cast<float>(size * 2.0f);
    for (int i = 0; i < size * size; ++i) {
        if (C[i] != correct_value) {
            std::fprintf(stderr, "Verification failed at index %d: expected %.2f, got %.2f\n",
                         i, correct_value, C[i]);
            return false;
        }
    }

    return true;
}



// Read this to know what the benchmark is doing:
void
benchmark_kernel(const KernelEntry &kernel)
{ 
    // make sure that the kernel implemenation is correct
    if (!verify_kernel(kernel.fn)) {
        std::fprintf(stderr, "Verification failed for %.*s, skipping benchmark.\n",
                     static_cast<int>(kernel.name.size()), kernel.name.data());
        return;
    }

    std::printf("\n=== Kernel: %.*s ===\n",
                static_cast<int>(kernel.name.size()), kernel.name.data());
    std::printf("%8s  %20s  %12s  %12s\n", "Size", "Avg TSC Cycles", "GFLOP/s", "GMAC/s");
    
    // test performance at each size
    for (int size : MATRIX_SIZES) {
        const auto A = std::vector<std::bfloat16_t>(size * size, 1.0bf16);
        const auto raw_B = std::vector<std::bfloat16_t>(size * size, 1.5bf16);
        const auto B = pack_vnni_b(raw_B.data(), size, size);
        auto C = std::vector<float>(size * size);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            kernel.fn(A.data(), B.data(), C.data(), size, size, size);
        }

        // Benchmark
        const auto start_time = std::chrono::steady_clock::now();
        _mm_lfence();
        const std::uint64_t start_cycles = __rdtsc();
        _mm_lfence();

        for (int i = 0; i < MEASURE_ITERS; ++i) {
            kernel.fn(A.data(), B.data(), C.data(), size, size, size);
        }

        unsigned int aux;
        const std::uint64_t end_cycles = __rdtscp(&aux);
        _mm_lfence();
        const auto end_time = std::chrono::steady_clock::now();

        const double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        const double avg_cycles = static_cast<double>(end_cycles - start_cycles) / MEASURE_ITERS;
        const double total_flops = 2.0 * size * size * size * MEASURE_ITERS;
        const double gflops = (total_flops / elapsed_sec) / 1e9;
        const double gmacs = gflops / 2.0;

        std::printf("%4d^3    %20.2f  %12.2f  %12.2f\n",
                    size, avg_cycles, gflops, gmacs);
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
        std::printf("Usage: %s [--all|--none|--low|--med|--opt]\n", argv[0]);
        return 0;
    }

    bool matched = false;
    for (const auto &kernel : KERNELS) {
        if (target_flag == "--all" || target_flag == kernel.flag) {
            benchmark_kernel(kernel);
            matched = true;
        }
    }

    if (!matched) {
        std::fprintf(stderr, "Unknown option: %s\n", argv[1]);
        std::fprintf(stderr, "Usage: %s [--all|--none|--low|--med|--opt]\n", argv[0]);
        return 1;
    }

    return 0;
}
