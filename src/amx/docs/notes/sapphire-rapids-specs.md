# Sapphire Rapids & Golden Cove Architecture Specifications

## Golden Cove Core Microarchitecture

* **Allocation Width:** 6-wide allocation
* **Execution Ports:** 12 execution ports
* **Retirement Width:** 8-wide retirement
* **Instruction Fetch / Decode:** Wider fetch with 6 decoders
* **Load Bandwidth:** Max load bandwidth of 3 loads per cycle



## Cache & Memory Subsystem

* **General Load Bandwidth:** Maximum load bandwidth increased from 2 to 3 loads per cycle.
* **Vector & Matrix Load Bandwidth:** Bandwidth for Intel AVX-512 loads, Intel AMX loads, and MMX/x87 loads remains at a maximum of 2 loads per cycle.



## Sapphire Rapids Platform Specifications

### Memory Support
* **1 DPC (DIMM Per Channel):** Up to DDR5-4800 MT/s
* **2 DPC (DIMMs Per Channel):** Up to DDR5-4400 MT/s

### Operating Frequencies
* **Max Turbo Frequency:** 3.80 GHz
* **Processor Base Frequency:** 2.10 GHz



## Intel AMX Frequency Licensing & Thermal Budgets

* **Frequency License Levels:** Intel AMX introduces a distinct frequency license level. Its nominal maximum frequency is lower than that of the Intel AVX-512 license level due to the high power density and thermal budget required when driving the TMUL (Tile Matrix Multiply) units.
* **Light Workload Turbo:** When Intel AMX unit utilization is lower than 15%, the processor may exceed the nominal max frequency associated with the Intel AMX license.

> Single-core AMX turbo seemes to be ~2.9 GHz; all-core turbo with Intel AMX operates around 2.1 GHz.



## Intel Performance Metric Definitions

- **Throughput:** The number of clock cycles required to wait before the issue ports are free to accept the same instruction again. This number can be lower than 1, e.g., 0.5 or 0.33, indicating that multiple instructions could be executed in parallel in a given cycle. For instructions that execute at allocation, 1/alloc_width (1/6 for Golden Cove microarchitecture) was used as throughput in the tables. 
- **Latency:** The number of clock cycles that are required for the CPU to complete the execution of all of the µops that form an instruction. 
