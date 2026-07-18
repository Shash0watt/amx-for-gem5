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

    // -> You need a engine and a stream to use oneDNNL

    // Setup engine & stream
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    // -> Then you can create the data that you are going to load into oneDNNL

    // first specify the dimensions for the 'tensor'
    const memory::dim M = state.range(0);
    const memory::dim K = state.range(0);
    const memory::dim N = state.range(0);

    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims dst_dims = {M, N};

    // then create a buffer normally to hold our data
    std::vector<float> src_data(M * K);
    std::vector<float> weights_data(K * N);
    std::vector<float> dst_data(M * N);

    // we can fill the buffer with our data
    std::generate(src_data.begin(), src_data.end(),
                  [i = 0]() mutable { return std::cos(i++ / 10.f); });

    std::generate(weights_data.begin(), weights_data.end(),
                  [i = 0]() mutable { return std::sin(i++ * 2.f); });

    // -> Now we need to create the descriptors for oneDNN and use them

    // these are the memory descriptors for our tensors
    auto src_md =
        memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ab);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                   memory::format_tag::ab);
    auto dst_md =
        memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ab);

    // we can then allocate the 'space' on the engine using the descriptors
    // and write our initial data into the oneDNN memory structure
    auto src_mem = memory(src_md, engine, src_data.data());
    auto weights_mem = memory(weights_md, engine, weights_data.data());
    auto dst_mem = memory(dst_md, engine, dst_data.data());

    // Instead of calling a generic function, oneDNN looks at
    // the metadata and the engine's capabilites and compiles the assembly for
    // it Normally in the place of memory::desc() we would have a 'bias'
    // tensor, but this bypasses it
    auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md,
                                            memory::desc(), dst_md);
    auto matmul_prim = matmul(matmul_pd);

    // the primitve knows how to multiply but not which ones to multiply
    // the matmul take a src(A), weights(B), a bias (C) (which we have removed)
    // and the destination and does dest = A*B+C
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // benchmark loop
    double flops_per_iteration = 2.0 * double(M) * double(K) * double(N);
    for (auto _ : state) {
        // execute the matmul
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