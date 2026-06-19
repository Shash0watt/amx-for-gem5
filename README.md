#
Research project implementing Intel AMX features into gem5 to profile matrix multiplication and LLM inference workloads.

## Usage

### 1. Compile C++ Source

``` bash
cd src && make
```

### 2. Build gem5 & m5ops

Run these commands from the root /gem5 directory:

gem5: ` scons build/{ISA}/gem5.{variant} -j {cpus} `

m5ops: ba`` scons build/{TARGET_ISA}/out/m5 ` (run from /gem5/util/m5)

### 3. Run Simulation

``` bash
./path-to-gem5-build -rs amx/tb.py
```
