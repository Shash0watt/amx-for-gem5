amx-for-gem5
Research project in implementing some AMX features into gem5 for profiling matrix multiplication/LLM inference workloads

Usage Compile C++ using 'make' in src dir Compile changes to gem5 using 'scons build/{ISA}/gem5.{variant} -j {cpus}' in /gem5 Compile change to m5ops using 'scons build/{TARGET_ISA}/out/m5' in /gem5/util/m5

Run sim using './path-to-gem5-build -rs amx/tb.py'
