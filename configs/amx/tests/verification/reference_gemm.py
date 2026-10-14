#!/usr/bin/env python3
# Plain GEMM on the same inputs as intel_manual_{gem5,amx}_GEMM.cpp, accumulated
# in FP32 in the same order as AMX, so the output should match bit-for-bit.
# Writes C as raw FP32 hex to correct.out.

import struct

M, N, K = 16, 512, 2048
VALUES = [0.0, 1.0, -2.0, 3.0, 0.5, -1.25, 0.1, 2.75]


def fp32_bits(x):
    return struct.unpack("I", struct.pack("f", x))[0]


def from_bits(bits):
    return struct.unpack("f", struct.pack("I", bits))[0]


def to_bf16(x):
    # Keep the top 16 bits of the FP32 value, rounding to nearest even.
    bits = fp32_bits(x)
    return from_bits((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000)


# These inputs repeat every eight rows of A and eight columns of B, so C
# repeats in both dimensions. Compute each distinct dot product once, keeping
# all K accumulation steps and their FP32 rounding unchanged.
period = len(VALUES)
A = [[to_bf16(VALUES[(m + k) % period]) for k in range(K)]
     for m in range(period)]
B = [[to_bf16(VALUES[(3 * k + n) % period]) for n in range(period)]
     for k in range(K)]

C = []
for m in range(period):
    row = []
    for n in range(period):
        acc = 0.0
        for k in range(K):
            acc = from_bits(fp32_bits(acc + A[m][k] * B[k][n]))  # round to FP32
        row.append(f"{fp32_bits(acc):08x}")
    C.append(row)

with open("correct.out", "w") as f:
    for m in range(M):
        f.write(" ".join(C[m % period][n % period] for n in range(N)) + "\n")
