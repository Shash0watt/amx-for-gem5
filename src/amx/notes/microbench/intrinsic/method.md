## Measuring core cycles

RDTSC (read time-stamp-counter) is the CPU timestamp counter every clock cycle (and resets it to 0 whenever the processor is reset)
Modern intel processors have an invariant TSC (it does not scale with clock frequency) - so this ends up being more like a time taken..

There are some options to make this more accurate 
1. lock down the clock fequency and use tsc
2. use RDPMC (read performance-monitoring-counters)
    - CPU_CLK_UNHALTED.THREAD is a ctr that tracks the cycles your code took
    - but it needs user space permissions

3. use Linux perf counters (perf_event_open)
    - does NOT need root / sudo permissions
    - PERF_COUNT_HW_CPU_CYCLES counts actual unhalted core cycles not the invariant TSC
    - can ignore kernel cycles (`exclude_kernel = 1`) to only measure userspace

### using perf_event_open
to use it in C / C++:
1. we set up struct perf_event_attr 
2. call syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0) to attach to the calling thread
3. read cycle counts with read(fd, &count, sizeof(count))
4. close the fd when done with close(fd)

syscall details, hardware event attributes, and examples of perf_event_open uses: \
https://man7.org/linux/man-pages/man2/perf_event_open.2.html
