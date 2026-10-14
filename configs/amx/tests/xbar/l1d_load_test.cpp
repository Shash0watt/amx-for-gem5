// CPU-only version of xbar_load_test: a burst of 16 L1D-hit loads, for
// measuring how fast the L1D serves a CPU with no xbar in between.
// Run with configs/amx/tests/xbar/l1d_only_config.py.

#include <cstdint>

#include <gem5/m5ops.h>

// 16 independent loads, one per cache line.
static void
cpuLoadBurst(const uint8_t *lines)
{
    asm volatile(
        ".irp off,0,64,128,192,256,320,384,448,"
        "512,576,640,704,768,832,896,960\n"
        "mov \\off(%0), %%rax\n"
        ".endr\n"
        :
        : "r"(lines)
        : "rax", "memory");
}

int
main()
{
    alignas(64) static uint8_t cpuLines[16][64];

    // Touch every line first so the timed loads all hit in the L1D.
    for (int row = 0; row < 16; ++row) {
        cpuLines[row][0] = row;
    }

    m5_work_begin(0, 0);
    cpuLoadBurst(&cpuLines[0][0]);
    m5_work_end(0, 0);

    return 0;
}
