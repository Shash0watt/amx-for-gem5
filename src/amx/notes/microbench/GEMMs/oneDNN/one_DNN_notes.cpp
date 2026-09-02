#include <cstdio>
#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

int
main()
{
    // -> You need a engine and a stream to use oneDNNL

    // Setup engine & stream
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    // -> Then you can create the data that you are going to load into oneDNNL

    // first specify the dimensions for the 'tensor'
    const memory::dim M = 4;
    const memory::dim K = 4;
    const memory::dim N = 4;

    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims dst_dims = {M, N};

    // then create a buffer normally to hold our data
    // Fill src with 1s and weights with 1.5s
    std::vector<float> src_data(M * K, 1.0f);
    std::vector<float> weights_data(K * N, 1.5f);
    std::vector<float> dst_data(M * N, 0.0f);

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

    // execute the matmul
    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    // print the results
    std::printf("Result Matrix (4x4, expected %.1f per element):\n",
                static_cast<float>(K * 1.0f * 1.5f));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::printf("%6.2f ", dst_data[i * N + j]);
        }
        std::printf("\n");
    }

    return 0;
}