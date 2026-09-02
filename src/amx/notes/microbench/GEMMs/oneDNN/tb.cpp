#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdfloat>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

#include "oneapi/dnnl/dnnl.hpp"

// Compile:
//   g++ -std=c++23 -O3 tb.cpp -ldnnl -o tb
//
// Usage:
//   ./tb

constexpr int WARMUP_ITERS = 20;
constexpr int MEASURE_ITERS = 100;
constexpr int MATRIX_SIZES[] = {32, 64, 128, 256, 512, 1024};
constexpr int MAX_SIZE = 1024;

// 64-byte aligned static memory buffers to prevent AMX cache line split stalls
alignas(64) static std::bfloat16_t src_data[MAX_SIZE * MAX_SIZE];
alignas(64) static std::bfloat16_t weights_data[MAX_SIZE * MAX_SIZE];
alignas(64) static float dst_data[MAX_SIZE * MAX_SIZE];

using namespace dnnl;

bool
verify_matmul(dnnl::engine &engine, dnnl::stream &engine_stream, int size = 64)
{
    const memory::dim M = size;
    const memory::dim K = size;
    const memory::dim N = size;

    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims dst_dims = {M, N};

    std::fill_n(src_data, M * K, 1.0bf16);
    std::fill_n(weights_data, K * N, 2.0bf16);
    std::fill_n(dst_data, M * N, 0.0f);

    auto src_md = memory::desc(src_dims, memory::data_type::bf16, memory::format_tag::ab);
    auto weights_md_any = memory::desc(weights_dims, memory::data_type::bf16, memory::format_tag::any);
    auto weights_md_user = memory::desc(weights_dims, memory::data_type::bf16, memory::format_tag::ab);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ab);

    // Pre-allocate user scratchpad to avoid runtime allocations
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);

    auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md_any, memory::desc(), dst_md, attr);
    auto matmul_prim = matmul(matmul_pd);

    auto src_mem = memory(src_md, engine, src_data);
    auto weights_user_mem = memory(weights_md_user, engine, weights_data);
    auto dst_mem = memory(dst_md, engine, dst_data);

    auto weights_mem = weights_user_mem;
    if (matmul_pd.weights_desc() != weights_md_user) {
        weights_mem = memory(matmul_pd.weights_desc(), engine);
        reorder(weights_user_mem, weights_mem).execute(engine_stream, weights_user_mem, weights_mem);
        engine_stream.wait();
    }

    auto scratchpad_mem = memory(matmul_pd.scratchpad_desc(), engine);

    std::unordered_map<int, memory> matmul_args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_SCRATCHPAD, scratchpad_mem}
    };

    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    const float correct_value = static_cast<float>(size * 2.0f);
    for (int i = 0; i < size * size; ++i) {
        if (std::abs(dst_data[i] - correct_value) > 1e-2f) {
            std::fprintf(stderr, "Verification failed at index %d: expected %.2f, got %.2f\n",
                         i, correct_value, dst_data[i]);
            return false;
        }
    }

    return true;
}

void
benchmark_oneDNN()
{
    // 1. Setup execution engine (CPU) and execution stream
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    if (!verify_matmul(engine, engine_stream)) {
        std::fprintf(stderr, "Verification failed, skipping benchmark.\n");
        return;
    }

    std::printf("\n=== oneDNN BF16 GEMM ===\n");
    std::printf("%8s  %20s  %12s\n", "Size", "Avg TSC Cycles", "GFLOP/s");

    for (int size : MATRIX_SIZES) {
        // 2. Define matrix dimensions (M x K * K x N = M x N)
        const memory::dim M = size;
        const memory::dim K = size;
        const memory::dim N = size;

        memory::dims src_dims = {M, K};
        memory::dims weights_dims = {K, N};
        memory::dims dst_dims = {M, N};

        std::fill_n(src_data, M * K, 1.0bf16);
        std::fill_n(weights_data, K * N, 1.5bf16);
        std::fill_n(dst_data, M * N, 0.0f);

        // Memory descriptors (shape, data type, and layout)
        auto src_md = memory::desc(src_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto weights_md_any = memory::desc(weights_dims, memory::data_type::bf16, memory::format_tag::any);
        auto weights_md_user = memory::desc(weights_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ab);

        // Configure user scratchpad to eliminate dynamic allocation in benchmark loop
        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);

        // Primitive descriptor and primitive creation
        auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md_any, memory::desc(), dst_md, attr);
        auto matmul_prim = matmul(matmul_pd);

        // Wrap 64-byte aligned memory buffers into oneDNN memory objects
        auto src_mem = memory(src_md, engine, src_data);
        auto weights_user_mem = memory(weights_md_user, engine, weights_data);
        auto dst_mem = memory(dst_md, engine, dst_data);

        // Pre-pack weights into optimal AMX format once outside the timing loop
        auto weights_mem = weights_user_mem;
        if (matmul_pd.weights_desc() != weights_md_user) {
            weights_mem = memory(matmul_pd.weights_desc(), engine);
            reorder(weights_user_mem, weights_mem).execute(engine_stream, weights_user_mem, weights_mem);
            engine_stream.wait();
        }

        // Allocate scratchpad buffer once outside the timing loop
        auto scratchpad_mem = memory(matmul_pd.scratchpad_desc(), engine);

        // Bind tensor arguments for execution (SRC: A, WEIGHTS: B, DST: C, SCRATCHPAD)
        std::unordered_map<int, memory> matmul_args = {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCRATCHPAD, scratchpad_mem}
        };

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            matmul_prim.execute(engine_stream, matmul_args);
            engine_stream.wait();
        }

        // Benchmark
        const auto start_time = std::chrono::steady_clock::now();
        _mm_lfence();
        const std::uint64_t start_cycles = __rdtsc();
        _mm_lfence();

        for (int i = 0; i < MEASURE_ITERS; ++i) {
            matmul_prim.execute(engine_stream, matmul_args);
            engine_stream.wait();
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
main()
{
    benchmark_oneDNN();
    return 0;
}
