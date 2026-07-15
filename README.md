# AMX for gem5
Research project implementing Intel AMX features into gem5 to profile matrix multiplication and LLM inference workloads.

## Repository Structure

```text
amx-for-gem5/
├── configs/
│   └── amx/ (Simulation config, C++ test workloads)
├── src/
│   ├── amx/ (Custom Intel AMX accelerator SimObject)
│   ├── cpu/
│   │   ├── o3/
│   │   │   └── lsq.cc (Modified LSQ to intercept AMX packets)
│   │   └── base.cc (Modified to interface with the AMX accelerator)
│   └── sim/
│       └── pseudo_inst.cc (Custom m5ops for AMX intrinsics)
└── README.md
```

## Usage

### 1. Compile Workloads
```bash
make
```

### 2. Build gem5 & m5ops
```bash
# Build gem5 (Replace {cpus} with your cpu core count)
scons build/X86/gem5.opt -j {cpus}

# Build m5ops utility
cd util/m5 && scons build/x86/out/m5
```

### 3. Run Simulation
```bash
./build/X86/gem5.opt -rs configs/amx/tb.py
```

## Developer Utilities

* **Syntax-only linting build:**
  ```bash
  scons build/X86/gem5.opt CCFLAGS="-fsyntax-only"
  ```
* **Check logged-in users:**
  ```bash
  w
  ```
* **CPU and memory usage summary per user:**
  ```bash
  ps aux | awk '{arr[$1]+=$3; arr2[$1]+=$4} END {for (i in arr) print i, "CPU%:", arr[i], "MEM%:", arr2[i]}' | sort -nk3
  ```
