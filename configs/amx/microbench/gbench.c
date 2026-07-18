#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

// Google Benchmark and oneDNN Headers
#include <benchmark/benchmark.h>
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

static void
oneDNN_matmul(benchmark::State &state)
{
    // Setup engine & stream
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    const memory::dim M = state.range(0);
    const memory::dim K = state.range(0);
    const memory::dim N = state.range(0);

    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims dst_dims = {M, N};

    // Keep host data initialization in standard f32 for easy generation
    std::vector<float> src_f32_data(M * K);
    std::vector<float> weights_f32_data(K * N);

    std::generate(src_f32_data.begin(), src_f32_data.end(),
                  [i = 0]() mutable { return std::cos(i++ / 10.f); });

    std::generate(weights_f32_data.begin(), weights_f32_data.end(),
                  [i = 0]() mutable { return std::sin(i++ * 2.f); });

    // 1. Create descriptors for your raw, unoptimized host data (f32,
    // row-major)
    auto src_f32_md =
        memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ab);
    auto weights_f32_md = memory::desc(weights_dims, memory::data_type::f32,
                                       memory::format_tag::ab);

    auto src_f32_mem = memory(src_f32_md, engine, src_f32_data.data());
    auto weights_f32_mem =
        memory(weights_f32_md, engine, weights_f32_data.data());

    // 2. Create descriptors for the target BF16 computation
    // Changing format_tag::ab -> format_tag::any allows oneDNN to choose
    // the absolute best internal blocked layout for Intel AMX hardware.
    auto src_bf16_md = memory::desc(src_dims, memory::data_type::bf16,
                                    memory::format_tag::any);
    auto weights_bf16_md = memory::desc(weights_dims, memory::data_type::bf16,
                                        memory::format_tag::any);
    auto dst_bf16_md = memory::desc(dst_dims, memory::data_type::bf16,
                                    memory::format_tag::any);

    // Create the primitive descriptor using our target configuration
    auto matmul_pd = matmul::primitive_desc(
        engine, src_bf16_md, weights_bf16_md, memory::desc(), dst_bf16_md);
    auto matmul_prim = matmul(matmul_pd);

    // 3. Allocate actual engine spaces using the exact layout shapes chosen by
    // the primitive
    auto src_bf16_mem = memory(matmul_pd.src_desc(), engine);
    auto weights_bf16_mem = memory(matmul_pd.weights_desc(), engine);
    auto dst_bf16_mem = memory(matmul_pd.dst_desc(), engine);

    // 4. Reorder (downcast + pack) your plain f32 data into the optimized AMX
    // layouts. Crucially, we execute this OUTSIDE the loop so it won't corrupt
    // our timed benchmark metrics!
    reorder(src_f32_mem, src_bf16_mem)
        .execute(engine_stream, src_f32_mem, src_bf16_mem);
    reorder(weights_f32_mem, weights_bf16_mem)
        .execute(engine_stream, weights_f32_mem, weights_bf16_mem);
    engine_stream.wait();

    // Map arguments
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_bf16_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_bf16_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_bf16_mem});

    // 5. WARMUP RUN: Triggers the internal assembler to generate the JIT AMX
    // kernel right now. This wipes out the "first-run penalty" so your Google
    // Benchmark starts with a hot cache.
    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    // Benchmark loop
    double flops_per_iteration = 2.0 * double(M) * double(K) * double(N);
    for (auto _ : state) {
        matmul_prim.execute(engine_stream, matmul_args);
        engine_stream.wait();
    }

    // Google benchmark feature to show the processing rate
    state.counters["GFLOPS"] = benchmark::Counter(
        flops_per_iteration / 1e9,
        benchmark::Counter::kAvgThreads | benchmark::Counter::kIsRate);
}

BENCHMARK(oneDNN_matmul)->RangeMultiplier(2)->Range(128, 1024);

BENCHMARK_MAIN();