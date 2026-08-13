#include <chrono>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <string_view>

#include "amx_helpers.hh"
#include "gemms.hh"

// Compile from this directory:
// g++ -std=c++23 -O3 -mamx-tile -mamx-bf16 gemm_tb.cpp -o gemm_tb
//
// Select the GEMM implementation. Each run benchmarks square matrices with
// dimensions 128, 256, 512, and 1024:
// ./gemm_tb --none
// ./gemm_tb --low
// ./gemm_tb --med

constexpr int warmup_iterations = 100;
constexpr int measured_iterations = 100;
constexpr int matrix_sizes[] = {128, 256, 512, 1024};

using GemmFunction = void (*)(const std::bfloat16_t *,
                              const std::bfloat16_t *, float *, int, int, int);

struct BenchmarkResult
{
    double average_cycles;
    double gflops;
};

void
print_usage(const char *program_name)
{
    std::fprintf(stderr, "Usage: %s --none|--low|--med\n", program_name);
}

BenchmarkResult
measure_gemm(GemmFunction gemm, int size)
{
    const auto A = make_bf16_matrix(size, size, 256.5bf16);
    const auto unpacked_B = make_bf16_matrix(size, size, 128.25bf16);
    const auto B = pack_bf16_matrix(unpacked_B.data(), size, size);
    auto C = std::vector<float>(size * size);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        gemm(A.data(), B.data(), C.data(), size, size, size);
    }

    const auto start_time = std::chrono::steady_clock::now();

    _mm_lfence();
    const std::uint64_t start_cycles = __rdtsc();
    _mm_lfence();

    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
        gemm(A.data(), B.data(), C.data(), size, size, size);
    }

    unsigned int tsc_aux;
    const std::uint64_t end_cycles = __rdtscp(&tsc_aux);
    _mm_lfence();

    const auto end_time = std::chrono::steady_clock::now();

    const double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    const double total_operations =
        2.0 * size * size * size * measured_iterations;

    BenchmarkResult result;
    result.average_cycles =
        static_cast<double>(end_cycles - start_cycles) / measured_iterations;
    result.gflops = total_operations / elapsed_seconds / 1.0e9;
    return result;
}

int
main(int argc, char **argv)
{
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Select the GEMM implementation from the command-line option.
    const std::string_view option = argv[1];
    GemmFunction selected_gemm;
    const char *selected_gemm_name;

    if (option == "--none") {
        selected_gemm = bf16_gemm_none;
        selected_gemm_name = "bf16_gemm_none";
    } else if (option == "--low") {
        selected_gemm = bf16_gemm_low;
        selected_gemm_name = "bf16_gemm_low";
    } else if (option == "--med") {
        selected_gemm = bf16_gemm_med;
        selected_gemm_name = "bf16_gemm_med";
    } else {
        std::fprintf(stderr, "Unknown kernel: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    if (!set_tiledata_use()) {
        std::fputs("Failed to request XTILEDATA permission.\n", stderr);
        return 1;
    }

    std::printf("Kernel: %s\n", selected_gemm_name);
    std::printf("%8s  %20s  %12s\n", "Size", "Average TSC cycles",
                "GFLOP/s");

    for (int size : matrix_sizes) {
        const BenchmarkResult result = measure_gemm(selected_gemm, size);
        std::printf("%4d^3    %20.2f  %12.2f\n", size,
                    result.average_cycles, result.gflops);
    }

    return 0;
}
