#pragma once

#include <stdfloat>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

inline std::vector<std::bfloat16_t>
make_bf16_matrix(int rows, int columns, std::bfloat16_t value)
{
    return std::vector<std::bfloat16_t>(rows * columns, value);
}

// Pack row-major B[K][N] into B[K/2][N][2] for TDPBF16PS.
inline std::vector<std::bfloat16_t>
pack_bf16_matrix(const std::bfloat16_t *B, int K, int N)
{
    constexpr int kpack = 4 / sizeof(std::bfloat16_t);
    std::vector<std::bfloat16_t> packed_B(K * N);

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            packed_B[(k / kpack) * (N * kpack) + n * kpack + k % kpack] =
                B[k * N + n];
        }
    }

    return packed_B;
}

inline bool
set_tiledata_use()
{
    constexpr int arch_req_xcomp_perm = 0x1023;
    constexpr int xfeature_xtiledata = 18;

    return syscall(SYS_arch_prctl, arch_req_xcomp_perm,
                   xfeature_xtiledata) == 0;
}
