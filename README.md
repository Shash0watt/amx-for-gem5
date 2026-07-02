# AMX for gem5
Research project implementing Intel AMX features into gem5 to profile matrix multiplication and LLM inference workloads.

## Usage

### 1. Compile C++ Workloads

``` bash
make
```

### 2. Build gem5 & m5ops


i) gem5:
```bash
scons build/X86/gem5.opt -j {cpus}
```

ii) m5ops:
```bash
cd util/m5 && scons build/x86/out/m5
```

### 3. Run Simulation

``` bash
./build/X86/gem5.opt -rs configs/amx/tb.py
```


### (for me)

- for linting
``` bash
scons build/X86/gem5.opt CCFLAGS="-fsyntax-only"
```

- see users server
``` bash
w
```

- see summary of usage
``` bash
ps aux | awk '{arr[$1]+=$3; arr2[$1]+=$4} END {for (i in arr) print i, "CPU%:", arr[i], "MEM%:", arr2[i]}' | sort -nk3
```
