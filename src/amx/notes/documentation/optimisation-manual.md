# Key Takeaways on AMX from Chapter 20 of Intel's Optimization Reference Manual

## Detecting Intel AMX Support

### Using CPUID
In GCC and Clang, the header `<cpuid.h>` provides a wrapper macro to query processor capabilities:

```c
#include <cpuid.h>

unsigned int eax, ebx, ecx, edx;
__cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
```

This macro moves the leaf into EAX, the subleaf into ECX, executes the `cpuid` instruction, and writes the 128 bits of returned hardware information into the four output registers.

* **Leaf (EAX):** The main feature category or chapter you want to query (e.g., 0x7 for Structured Extended Features, 0x1D for AMX Tile Palettes).
* **Subleaf (ECX):** The sub-page within that category. It is used when a single leaf has more data than can fit in a single 128-bit response.

### AMX Detection

Verifying full AMX support requires three checks:

#### 1. Check Hardware Support (Leaf 0x7, Subleaf 0)
Query `__cpuid_count(7, 0, eax, ebx, ecx, edx)` and inspect EDX. All three base feature bits must be 1:
* **Bit 22 (AMX-BF16):** BFloat16 matrix multiplication instructions.
* **Bit 24 (AMX-TILE):** Base tile architecture and configuration registers (`ldtilecfg`, `sttilecfg`, `tileloadd`, `tilestored`).
* **Bit 25 (AMX-INT8):** 8-bit integer matrix multiplication instructions.

#### 2. Check OS State Management (CPUID Leaf 1 & XGETBV)
The OS must be capable of saving and restoring AMX state across context switches:
1. Query CPUID Leaf 1 to ensure ECX bit 26 (OSXSAVE) is set to 1.
2. Query `_xgetbv(0)` to ensure XCR0 bit 17 (XTILECFG) and bit 18 (XTILEDATA) are both enabled by the kernel.

#### 3. Get Tile Geometry (Leaf 0x1D, Subleaf 1)
Query Palette 1 implementation limits and unpack the 16-bit packed fields in EBX and ECX:

```c
unsigned int eax, ebx, ecx, edx;
__cpuid_count(0x1D, 1, eax, ebx, ecx, edx);

int total_tile_bytes  = eax;                  // 8192 bytes total
int bytes_per_tile    = (ebx >> 16) & 0xffff; // 1024 bytes per tile
int max_bytes_per_row = ebx & 0xffff;         // 64 bytes (512 bits) per row
int max_tiles         = (ecx >> 16) & 0xffff; // 8 tile registers (TMM0–TMM7)
int max_rows          = ecx & 0xffff;         // 16 rows per tile
```

Querying Leaf 0x1D at runtime isn't really necessary. Current Intel AMX implementations (Sapphire Rapids, Emerald Rapids, Granite Rapids) use a fixed Palette 1 configuration: exactly 8 tile registers (TMM0–TMM7), 16 rows per tile, and 64 bytes per row (1024 bytes per tile, 8 KB total).

### Actually Using Intel AMX (Linux Syscalls)

Even on hardware and kernels that support AMX, executing an AMX instruction will immediately trigger an Invalid Opcode fault (#UD / SIGILL) unless the application requests permission from the Linux kernel for the calling thread:

```c
#include <stdbool.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

bool enable_amx(void) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        return false; // Permission denied or kernel lacks dynamic xstate support
    }
    return true;
}
```

 
## Intel AMX Frequencies
When the Intel AMX unit utilization is lower than 15%, the processor may exceed the nominal max frequency associated with the Intel AMX license. Otherwise Intel AMX adds yet another license level whose max frequency is usually lower than that of the Intel AVX-512 license.

## Intel AMX Throuput and Latency
*Check the table in the PDF*

## Data Strucutre Alignment
Input and output data structures for GEMM and Convolutions must be 64-byte aligned (cache-line aligned). Aligning to larger boundaries (128-byte, 256-byte) should be avoided.

## Tiles in Intel AMX
- 8 2D tile registers (tmm0–tmm7), up to 1 KB each.
- when looking at OPCODE C A B:
    - A-tiles can have between 1-16 rows and 1-MAX_TILE_K columns.
    - B-tiles can have between 1-MAX_TILE_K rows and 1–16 columns.
    - C-tiles can have between 1-16 rows and 1–16 columns.

 MAX_TILE_K=64/sizeof(type_t), and type_t is the data type being operated on. Therefore, MAX_TILE_K=64 for
(u)int8 data, and MAX_TILE_K=32 for bfloat16 data.

### Matrix B Layout
The B matrix must undergo a re-layout before it can be used within the
corresponding Intel AMX multiply instruction. Matrix B requires re-layout into VNNI format (vertical K-packing into Dwords)

*Check the algo in the PDF*


## Optimisations

### General Optimisations:
- **Minize tile Loads:**
    - Keep the K loop outside the allocation of accumaltors to reduce cache pressure
    - Use 2D accumalators to maximise reuse
- **Software Pipelining of Tile Loads and Stores:**
    - Interleave memory load/store instructions with TDP* compute instructions to avoid execution resource bottlenecks 
    - > this is a strong argument for an internal queue, and scoreboarding of instructions


### For larger sized matricies
- Cache Blocking 
    > For me to review in more detail rn
- Non-Temporal Tile Loads
    - Streaming large, single-use activation or weight matrices through L1 cache evicts smaller, frequently reused tiles.
    - Use non-temporal tile loads (TILELOADDT1) for the larger, non-reusable data

- **Store to Load Forwarding**
TILELOAD cannot forward data directly from CPU store buffers; it must wait until the store drains to the memory hierarchy/cache, incurring a heavy stall. Maintain a distance of several tens of cycles between regular stores and subsequent TILELOADs of the same memory address.
> hmm


# Hardware & Microarchitecture Insights for Modeling Intel AMX

- **Executes Synchronously and Contends for OoO Machine Resources** (Section 20.11.2)
  - Synchronous In-Core Execution: AMX instructions (LDTILECFG, TILELOADD, TDP, TILESTORED) are decoded, issued, and tracked within the core's standard Out-of-Order pipeline (ROB, RS, and LSQ). AMX is integrated into the core rather than acting as an asynchronous off-core accelerator.
  - Out-of-Order Resource Pressure: Large unpipelined matrix multiplication and vector post-processing blocks can exhaust core out-of-order execution resources (ROB/RS entries), which is why software pipelining/interleaving is required.

- **Pipelines TMUL and Memory Units with Fixed Latencies and Throughputs** (Section 20.3, Table 20-2)
  - Fixed Latency & Throughput Profile:
    - TDP*: Latency = 52 cycles, Throughput = 16 cycles.
    - This makes sense with their claim of 2048 int ops per second : 
    - 16 × 16 × 64 MACs possible with tiles of int8s
    - 2ops/MAC -? 16 × 16 × 64 × 2 = 32,768 INT8 operations
    - 32,768 operations / 16 cycle wait = 2048 INT8 ops/cycle

- **Stalls Tile Loads on Store Buffer Conflicts Without Direct Forwarding** (Section 20.15)
  - No Store-Buffer Forwarding to TILELOAD: Unlike scalar/vector loads, TILELOAD cannot bypass data directly from CPU store buffers. If an address overlap occurs with an in-flight store, hardware stalls until the store buffer drains to the L1 Data Cache (DCU).
  - Restricted Forwarding from TILESTORED: Forwarding from TILESTORED to regular scalar/vector loads is supported only for 64-byte (cache-line) aligned chunks and has performance outliers (requires tens of cycles separation).

- **Resolves WAR Dependencies to Allow Early Source Register Overwrites** (Sections 20.3 & 20.11, Examples 20-21/20-22)
  - Write-After-Read (WAR) Handling: As shown in optimized assembly listings (Examples 20-21 & 20-22), software can issue a TILELOADD into a source tile register (tmmX) immediately after a TDP* that reads tmmX, without waiting for the 52-cycle compute latency.
  - Microarchitectural Rationale: Because the core operates out-of-order and the TMUL execution pipeline consumes/reads source operands upon dispatch, source registers are freed for new loads as soon as operand access completes.


---
*Note: I haven't include some topics (convolutions, transpose kernels, OpenMP multithreading) aren't relvant to my sim implementation.*

