# AMX intrinsic results: hardware vs. gem5

Cycles per instruction, hardware medians. Latency for ops that
can't be chained (zero, load, store, config) = a loop of op + LFENCE, minus
the LFENCE on its own (MFENCE for stores; see Fences below). gem5 uses
amx_wait in place of the fence. "Intel doc" is the latency or
throughput from the optimization manual's table 20-2 (- = not given).

## TDPBF16PS (dot_product_test.cpp)

| Test                         | Hardware | gem5  | Intel doc |
| ---------------------------- | -------- | ----- | --------- |
| latency, chain through C     | 16.09    | 16.23 |           |
| latency, chain through A     | 52.05    | 52.19 | 52        |
| latency, chain through B     | 52.05    | 52.19 | 52        |
| throughput, six accumulators | 16.09    | 16.21 | 16        |

## TILEZERO (zero_test.cpp)

| Test                    | Hardware | gem5  | Intel doc |
| ----------------------- | -------- | ----- | --------- |
| latency                 | 0.00     | 15.49 | 16        |
| zero-to-use, A          | 0.00     | 16.00 |           |
| zero-to-use, C          | 9.00     | 16.00 |           |
| throughput, eight tiles | 0.32     | 9.30  | -         |

Zero-to-use = zero + TDPBF16PS + LFENCE minus TDPBF16PS + LFENCE (hardware
43.09 baseline; gem5 70.22 baseline, at 2.4 GHz).

## TILELOADD (load_test.cpp)

| Test                    | Hardware | gem5   | Intel doc |
| ----------------------- | -------- | ------ | --------- |
| latency, L1             | 23.54    | 24.06  | 45        |
| latency, L2             |          | 49.49  |           | 
| latency, DRAM           | 579.11   | 451.81 |           |
| throughput, eight tiles | 8.59     | 11.51  | 8         |

## TILESTORED (store_test.cpp)

| Test                    | Hardware | gem5  | Intel doc |
| ----------------------- | -------- | ----- | --------- |
| latency                 | 37.24    | 9.72  | -         |
| throughput, eight tiles | 16.27    | 16.45 | 16        |

## LDTILECFG (config_test.cpp)

| Test                            | Hardware | gem5   | Intel doc |
| ------------------------------- | -------- | ------ | --------- |
| latency                         | 198.00   | 203.66 | 204       |
| throughput, same config         | 199.59   | 204.03 |           |
| throughput, alternating configs | 202.10   | 204.03 |           |

## Fences

A loop of only the fence, no AMX op. Subtracted from "op + fence" to get the
latency. (gem5's amx_wait: 18.55.)

| Fence  | Hardware |
| ------ | -------- |
| LFENCE | 14.10    |
| MFENCE | 38.85    |

## GEMM kernel benchmarks

Cycles for one whole BF16 GEMM, kernels from GEMMs/amx/gemms.hh. Hardware:
median after 200 ms of warm-up, ~2.4 GHz (opt manual: 5 runs, 6acc: 21
runs). gem5 at 2.4 GHz. The opt manual C matches correct.out; the 6acc C
hashes differ from hardware for every size (not yet checked whether that is
rounding or wrong values).
% = (gem5 - hardware) / hardware.

| Algorithm  | M x N x K       | Hardware | gem5   | gem5 vs hardware |
| ---------- | --------------- | -------- | ------ | ---------------- |
| opt manual | 16 x 512 x 2048 | 130378   | 130533 | +0.1%            |
| 6acc       | 48 x 256 x 1024 | 33299    | 64246  | +92.9%           |
| 6acc       | 96 x 512 x 2048 | 478379   | 569113 | +19.0%           |
