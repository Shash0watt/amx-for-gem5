# AMX GEMM Optimization Handoff

## 1. Current Status & Benchmark Baseline

* **Current Baseline Kernel:** [`bf16_gemm_med_6acc_continuous_prefetch`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L518) in [gemms.hh](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh).
* **Architecture:** $3\times2$ Tile Grid (6 accumulators: TMM0–TMM5; 2 streaming tiles: TMM6 for B, TMM7 for A).
* **Matrix B Layout:** [`pack_continuous_b`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/helpers.hh#L75) (flat 1024-byte contiguous tiles with 64-byte row stride).
* **Prefetch Strategy:** Sparse 4-row sampling on matrix A (`prefetch_tile_sparse`) + header touching on matrix B (`prefetch_b_head`) + outer $N$-panel lookahead (`_MM_HINT_T1`).

### Performance vs oneDNN

| Matrix Size | Continuous Baseline (GFLOP/s) | Continuous + Prefetch (GFLOP/s) | oneDNN Target (GFLOP/s) | Gap at $256^3$ |
| :--- | :--- | :--- | :--- | :--- |
| **$32^3$** | 259.47 | 246.26 | 56.96 | +332% (Better warmup/dispatch) |
| **$64^3$** | 541.28 | 519.66 | 261.11 | +99% |
| **$128^3$** | 705.51 | 691.50 | 803.33 | -14% |
| **$256^3$** | 1069.02 | **1092.95** | **1457.32** | **-25% (Target focus)** |
| **$512^3$** | 456.58 | 486.55 | 569.08 | -15% |
| **$1024^3$** | 395.75 | 400.78 | 537.20 | -25% |

---

## 2. Post-Mortem: Why Batched Loads Failed (`bf16_gemm_2x2_pipelined`)

The experimental 2x2 batched-load kernel dropped performance from **1092 GFLOP/s to 966 GFLOP/s** at $256^3$.

### What Went Wrong
1. **Contradicts Hardware Execution Profile (Intel AMX Optimization Guide Ch. 20):**
   * `TDPBF16PS` (TMUL): **Latency = 52 cycles, Throughput = 16 cycles**.
   * `TILELOADD`: **Latency ≈ 20 cycles**.
   * When 4 `TILELOADD` instructions are issued in a continuous block, they contend for core Out-of-Order execution buffers (ROB, RS, LSQ) while the TMUL compute unit sits completely idle.
2. **Ignored Write-After-Read (WAR) Pipelining:**
   * In Intel AMX, `TILELOADD` can be issued into a source tile register **immediately after** a `TDP*` instruction reads from it, without waiting for the 52-cycle arithmetic completion.
   * Fine-grained **interleaving of `TILELOADD` and `TDP*`** is essential so TMUL arithmetic and memory loads execute in parallel.
3. **Lower Arithmetic Intensity:**
   * $3\times2$ grid: $\frac{6\text{ compute}}{5\text{ loads}} = \mathbf{1.20\text{ MACs/load}}$.
   * $2\times2$ grid: $\frac{4\text{ compute}}{4\text{ loads}} = \mathbf{1.00\text{ MACs/load}}$ (20% more load pressure).

---

## 3. Implemented Optimization Kernels

All optimizations have been integrated into individual feature kernels as well as a unified optimal kernel [`bf16_gemm_opt`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L1374) in [gemms.hh](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh):

| Optimization Feature | Kernel Function | CLI Flag | Layout | Microarchitectural Focus |
| :--- | :--- | :--- | :--- | :--- |
| **★ Unified Optimal GEMM** | [`bf16_gemm_opt`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L1374) | `--opt` | Continuous A & B | Combines 64B alignment + 3x2 grid + 2-way K-unrolling + balanced prefetch + L2 panel blocking ($NC=256$) |
| **1. 64B Alignment + Continuous A & B** | [`bf16_gemm_med_6acc_cont_ab_prefetch`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L692) | `--cont_ab` | Continuous A & B | Eliminates 8 KB strided memory jumps on Matrix A; converts loads to 1024B sequential cache bursts |
| **2. K-Loop Unroll by 2** | [`bf16_gemm_med_6acc_k_unroll2`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L831) | `--unroll2` | Continuous A & B | Unrolls $TK=64$; overlaps B load of sub-step 1 with compute of sub-step 0; cuts branch overhead by 50% |
| **3. True Double-Buffered 2x2** | [`bf16_gemm_2x2_double_buffered`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L1010) | `--db2x2` | Continuous A & B | Uses 4 input registers (TMM4–TMM7) to hide ~20-cycle `TILELOADD` latency behind TMUL compute |
| **4. L2 Cache Panel Blocking** | [`bf16_gemm_med_6acc_l2_blocked`](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh#L1107) | `--blocked` | Continuous A & B | Blocks $NC=256$ panel of B in L2 cache across full $M$ sweeps to minimize DRAM memory traffic |

---

## 4. Benchmark Harness & CLI Commands

Compile on an AMX-enabled system or gem5 simulation container:
```bash
cd src/amx/notes/microbench/GEMMs/amx
g++ -std=c++23 -O3 -mamx-tile -mamx-bf16 tb.cpp -o tb
```

Run benchmarks for individual optimizations:
```bash
./tb --all       # Benchmark all kernels
./tb --cont_ab   # Kernel 1 (Continuous A & B)
./tb --unroll2   # Kernel 2 (K-unroll by 2)
./tb --db2x2     # Kernel 3 (2x2 double buffered)
./tb --blocked   # Kernel 4 (L2 blocked)
```

---

## 5. Key References & Files

* **Kernel Implementations:** [gemms.hh](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/gemms.hh)
* **Packing & Alignment Helpers:** [helpers.hh](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/helpers.hh)
* **AMX Test Harness:** [tb.cpp](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/amx/tb.cpp)
* **oneDNN Benchmark Baseline:** [oneDNN/tb.cpp](file:///home/sjaguri/amx-for-gem5/src/amx/notes/microbench/GEMMs/oneDNN/tb.cpp)
* **Documentation:**
  * [optimisation-manual.md](file:///home/sjaguri/amx-for-gem5/src/amx/notes/documentation/optimisation-manual.md)
  * [optimised-gemm.md](file:///home/sjaguri/amx-for-gem5/src/amx/notes/docs/optimised-gemm.md)
  * `src/amx/notes/documentation/Intel AMX Optimisation Guide.pdf`


