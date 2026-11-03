
//   make all
//   ./build/X86/gem5.opt configs/amx/tests/xbar/load_cycles_config.py --cpu xbar
//   ./build/X86/gem5.opt configs/amx/tests/xbar/load_cycles_config.py --cpu no-xbar

#include <cstdint>
#include <gem5/m5ops.h>

alignas(64) static volatile uint64_t data[32];

// Makefile builds at -O0, at it seems like GCC ignores the unroll pragma,
// so O2 is needed for it to take effect. The loads don't depend on each
// other, so the O3 CPU should be able to issue them back to back.
__attribute__((optimize("O2"))) static void
loadBurst()
{
#pragma GCC unroll 32
    for (int i = 0; i < 32; ++i) {
        data[i];
    }
}

int
main()
{
    // warmup to put the data into the cache
    loadBurst();

    m5_work_begin(0, 0);
    loadBurst();
    m5_work_end(0, 0);

    return 0;
}
