// Helpers shared by the real-hardware AMX tests in this folder.

// How a test is measured, region() below:
// - The thread is pinned to one core (AMX_BENCH_CPU, or the current core).
// - Dot products run for 200 ms first, so the AMX frequency license settles.
// - A test runs `ops` instructions between two cycle-counter reads. Both
//   reads are fenced, so the region starts after earlier work has finished
//   and ends after the test's work has finished (gem5: amx_wait).
// - The test runs once untimed, then Repeats timed times. The output is the
//   min, median and max cycles per instruction.

// Cycles are core cycles in user mode, from a perf counter
// (perf_event_open, no root needed) read with RDPMC

#include <immintrin.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

inline void
fillConfig(TileConfig &config, int rows, int columnBytes)
{
    config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = rows;
        config.columnBytes[tile] = columnBytes;
    }
}

const int Repeats = 21;

// ---------------------------------------------------------------------------
// Reading the cycle counter.
// ---------------------------------------------------------------------------
inline int counterFd = -1;
inline volatile perf_event_mmap_page *counterPage = nullptr;
inline bool useRdpmc = false;
inline uint64_t
readCounter()
{
    // The kernel's protocol for reading the counter from user space (see
    // perf_event_mmap_page in linux/perf_event.h): retry if the kernel
    // changed the page while we read it.
    while (useRdpmc) {
        uint32_t sequence = counterPage->lock;
        asm volatile("" ::: "memory");
        uint32_t index = counterPage->index;
        int64_t offset = counterPage->offset;
        if (index == 0) {
            break; // not on the PMU right now
        }
        int shift = 64 - counterPage->pmc_width;
        int64_t raw = int64_t(__rdpmc(index - 1) << shift) >> shift;
        asm volatile("" ::: "memory");
        if (counterPage->lock == sequence) {
            return offset + raw;
        }
    }
    uint64_t count = 0;
    if (read(counterFd, &count, sizeof(count)) != sizeof(count)) {
        std::perror("reading the cycle counter");
        std::exit(1);
    }
    return count;
}

// Core cycles this thread has spent in user mode. The fences make the read
// wait for all earlier instructions, AMX included, and keep later ones from
// starting before it.
inline uint64_t
readCycles()
{
    _mm_lfence();
    uint64_t count = readCounter();
    _mm_lfence();
    return count;
}

inline uint64_t
nowNs()
{
    timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);
    return uint64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
}

// ---------------------------------------------------------------------------
// Setting up and measuring.
// ---------------------------------------------------------------------------

inline void
setUpBenchmark()
{
    // Ask Linux for permission to use the AMX tile registers.
    const int ArchReqXcompPerm = 0x1023;
    const int XfeatureXtiledata = 18;
    if (syscall(SYS_arch_prctl, ArchReqXcompPerm, XfeatureXtiledata) != 0) {
        std::perror("AMX permission request failed");
        std::exit(1);
    }

    // Pin to one core, so the counter and the caches stay on it.
    const char *cpuName = std::getenv("AMX_BENCH_CPU");
    int cpu = cpuName ? std::atoi(cpuName) : sched_getcpu();
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::perror("sched_setaffinity");
        std::exit(1);
    }

    // Count this thread's core cycles in user mode. Pinned, so the counter
    // always stays on the PMU instead of being time-shared with others.
    perf_event_attr attr = {};
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    attr.pinned = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    counterFd = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    if (counterFd < 0) {
        std::perror("perf_event_open failed (perf_event_paranoid must be <= 2)");
        std::exit(1);
    }
    void *page = mmap(nullptr, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED,
                      counterFd, 0);
    if (page != MAP_FAILED) {
        counterPage = static_cast<perf_event_mmap_page *>(page);
        useRdpmc = counterPage->cap_user_rdpmc;
    }

    // Run dot products for 200 ms so the AMX frequency license settles.
    alignas(64) static TileConfig config;
    alignas(64) static uint16_t ones[512];
    fillConfig(config, 16, 64);
    for (int i = 0; i < 512; ++i) {
        ones[i] = 0x3f80; // BF16 1.0
    }
    asm volatile("" ::: "memory"); // the intrinsics don't say they read memory
    _tile_loadconfig(&config);
    _tile_loadd(6, ones, 64);
    _tile_loadd(7, ones, 64);
    uint64_t startNs = nowNs();
    uint64_t startCycles = readCycles();
    while (nowNs() - startNs < 200000000) {
        for (int i = 0; i < 1000; ++i) {
            _tile_dpbf16ps(0, 6, 7);
            _tile_dpbf16ps(1, 6, 7);
            _tile_dpbf16ps(2, 6, 7);
            _tile_dpbf16ps(3, 6, 7);
        }
    }
    double ghz = double(readCycles() - startCycles) / double(nowNs() - startNs);

    // What one pair of counter reads costs. It is in every region, spread
    // over the region's instructions.
    uint64_t readCost = UINT64_MAX;
    for (int i = 0; i < 1000; ++i) {
        uint64_t start = readCycles();
        readCost = std::min(readCost, readCycles() - start);
    }

    std::printf("# core %d, counter read with %s, %lu cycles per read\n", cpu,
                useRdpmc ? "rdpmc" : "read() (rdpmc not allowed)", readCost);
    std::printf("# warm-up: %.2f GHz under AMX load\n", ghz);
    std::printf("%-46s %8s %8s %8s\n", "test (cycles per instruction)", "min",
                "median", "max");
    std::fflush(stdout);
}

// Time test(), which runs `ops` instructions, as one region (gem5: one work
// region), and print min, median and max cycles per instruction.
inline void
region(const char *name, int ops, void (*test)())
{
    test(); // warm up

    std::vector<double> cyclesPerOp;
    for (int repeat = 0; repeat < Repeats; ++repeat) {
        uint64_t start = readCycles();
        test();
        uint64_t end = readCycles();
        cyclesPerOp.push_back(double(end - start) / ops);
    }
    std::sort(cyclesPerOp.begin(), cyclesPerOp.end());
    std::printf("%-46s %8.2f %8.2f %8.2f\n", name, cyclesPerOp.front(),
                cyclesPerOp[Repeats / 2], cyclesPerOp.back());
    std::fflush(stdout);
}
