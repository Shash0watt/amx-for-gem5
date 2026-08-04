# Address Translation in gem5

This tutorial explains address translation in gem5 from the outside in. It
starts with what gem5 is trying to model, introduces the systems and programming
concepts used to build that model, and only then follows translation through
the gem5 source code and this repository's AMX accelerator.

The goal is not just to recognize a call to `translateTiming()`. By the end,
you should be able to:

- explain why address translation exists and what real hardware does;
- distinguish the simulated computer from the C++ program simulating it;
- explain how gem5 represents hardware components, time, and communication;
- follow a virtual address until it becomes a physical memory request;
- distinguish syscall-emulation translation from a full-system page walk;
- review an asynchronous translation implementation for correctness; and
- design a new translated memory requester in gem5.

## 1. What is gem5 trying to model?

gem5 is a program that models another computer. The modeled, or **simulated**,
computer can contain CPUs, MMUs, TLBs, caches, interconnects, memory controllers,
and physical memory. Software running inside that computer executes loads and
stores just as it would on hardware.

There are therefore two different computers to keep in mind:

```text
Host computer                       Simulated computer
-------------                       ------------------
Runs the gem5 C++ program           Runs the workload
Allocates C++ objects               Has CPUs and registers
Executes gem5 callbacks             Has virtual memory
Processes an event queue            Has caches and DRAM
Takes real wall-clock time          Advances in simulated time
```

If gem5 takes one second on the host to simulate ten microseconds of the target
machine, those are two different measurements. The host time tells us how fast
the simulator is. Simulated time tells us how the modeled machine behaves.

### 1.1 Behavior and timing are separate questions

For a memory load, gem5 may need to answer both:

1. **What data should the load return?**
2. **When should the data become available?**

A model can return the correct bytes while modeling timing incorrectly. For
example, directly reading memory may produce the correct value but skip the TLB,
page-table walker, caches, contention, and retries. That can be useful for a
debugger or program loader, but it is not an accurate model of a hardware load.

This distinction gives us three useful levels of correctness:

- **Functional correctness:** the simulated program sees the right result.
- **Architectural correctness:** visible behavior such as permissions and
  faults agrees with the architecture.
- **Timing correctness:** the operation uses the modeled resources and takes
  the intended simulated time.

When reviewing address translation, always ask which of these claims the code
is making.

## 2. The real system gem5 represents

At the highest level, a load follows this path:

```text
Program executes a load
          |
          v
CPU calculates a virtual address
          |
          v
MMU translates it to a physical address
          |
          v
Memory request enters the cache hierarchy
          |
          v
Data returns to the CPU
```

Address translation is one stage of the memory operation. It determines which
physical location the program is allowed to access. It does not itself fetch
the requested application data.

### 2.1 Why virtual addresses exist

A program normally uses **virtual addresses**. The operating system gives each
process its own virtual address space and maintains mappings from virtual pages
to physical pages.

Virtual memory provides several useful properties:

- two processes can use the same virtual address without using the same memory;
- a process cannot normally access another process's physical memory;
- virtual memory can be allocated without requiring physically contiguous
  memory; and
- each mapping can carry read, write, execute, and privilege permissions.

The address emitted by a load is therefore not usually the address used to
route the eventual memory-system request.

### 2.2 Pages and offsets

Virtual and physical address spaces are divided into fixed-size regions called
**pages**. With a 4 KiB page, the bottom 12 address bits are the page offset.

Consider the virtual address `0x1234`:

```text
Virtual address:       0x1234
Virtual page number:      0x1
Offset within page:      0x234
```

Suppose the page table maps virtual page `0x1` to physical page `0x9`:

```text
Physical page base:    0x9000
Page offset:           0x0234
                       ------
Physical address:      0x9234
```

Translation changes the page number but preserves the offset. This is the
central address calculation to keep in mind while reading the code.

### 2.3 Page tables

A **page table** stores virtual-to-physical mappings and their permissions. On
x86-64, the hardware normally uses a multi-level tree rather than one enormous
flat table. A control register identifies the top-level table, and fields from
the virtual address select entries at successive levels.

The important idea for this tutorial is not the exact bit layout. It is that a
page-table lookup may require several ordinary memory accesses before the
original load can proceed.

### 2.4 TLBs

A **translation lookaside buffer**, or TLB, caches recently used page-table
results.

```text
TLB hit                              TLB miss
-------                              --------
Mapping is already cached            Mapping is not cached
Translation finishes quickly         Page table must be consulted
No page-table walk is needed          Result may later be inserted in TLB
```

CPUs normally have an instruction TLB for instruction fetches and a data TLB
for loads and stores. gem5 commonly calls these the ITB and DTB.

### 2.5 Faults

Translation can fail. Examples include:

- the virtual page has no mapping;
- a load targets a non-readable page;
- a store targets a read-only page;
- user code targets a supervisor-only page; or
- an instruction fetch targets a non-executable page.

This failure is not a physical memory response. It is an architectural
**fault** discovered during translation. A faulting request must not continue
as if it had a valid physical address.

## 3. Concepts gem5 uses to build the model

gem5 represents hardware using C++ objects and advances simulated time through
events. Several general programming concepts appear repeatedly in translation
code.

### 3.1 State

Real hardware remembers information: TLB entries, outstanding requests, cache
tags, queue contents, and instruction status. A simulator stores equivalent
information in C++ objects.

For an asynchronous translation, useful state might include:

- the virtual request;
- which instruction initiated it;
- whether the translation was delayed;
- how many translations remain outstanding; and
- what to do when translation finishes.

If the operation can finish later, this state must remain valid until later.

### 3.2 Discrete-event simulation

gem5 is a discrete-event simulator. Components schedule work for particular
simulated ticks. The simulator repeatedly takes the next event from its event
queue, advances simulated time, and invokes the event.

An abstract TLB miss could look like this:

```text
Tick 100: CPU submits a translation
Tick 101: TLB detects a miss
Tick 102: page-table walk begins
Tick 140: final page-table entry returns
Tick 141: translation callback runs
Tick 142: data-cache request begins
```

The C++ call that starts an operation does not necessarily return its result.
Instead, the result can arrive through a callback during a later event.

### 3.3 Classes, inheritance, and virtual interfaces

gem5 uses base classes to define contracts shared by many implementations. For
example, `BaseMMU::Translation` says that a translation client must provide:

- `markDelayed()`, called when completion will happen later; and
- `finish(...)`, called when translation has completed or faulted.

Different CPU models and accelerators can implement these functions differently
while using the same MMU interface.

### 3.4 Callbacks and re-entrancy

A callback is an object or function that one component invokes to report a
result to another component. `translateTiming()` has an important property:
the callback may run **before `translateTiming()` returns**.

This can happen on a TLB hit and, in x86 syscall-emulation mode, on a miss that
gem5 resolves immediately from its emulated page table.

Consequently, code like this is unsafe:

```cpp
mmu->translateTiming(req, tc, callback, BaseMMU::Read);
outstandingTranslations++;
```

If `finish()` runs inside `translateTiming()`, it observes an outstanding count
that has not yet been incremented. The safe order is:

```cpp
outstandingTranslations++;
mmu->translateTiming(req, tc, callback, BaseMMU::Read);
```

This is a general asynchronous-programming rule: establish all state needed by
a completion handler before starting the operation.

### 3.5 Ownership and lifetime

C++ does not automatically know who owns every callback or packet. Translation
code must answer:

- Who deletes the callback?
- Who owns the `RequestPtr` while translation is outstanding?
- Who deletes a packet after its response arrives?
- Can the initiating instruction disappear before the callback?
- Can the `ThreadContext` disappear or change?

The `BaseMMU::Translation` contract explicitly permits a dynamically allocated
callback to clean itself up in `finish()`. Once `finish()` is called, the MMU
must treat that callback object as invalid.

### 3.6 State machines and counters

Asynchronous components commonly use states and counters instead of following
one uninterrupted call stack.

For example:

```text
PENDING
   |
   v
TRANSLATING -- one callback per fragment --> ACCESSING_MEMORY
                                                |
                                                v
                                            COMPLETED
```

If translations and memory requests overlap, one state name may not be enough.
Separate counters can express completion precisely:

```text
instruction is complete only if

    translation dispatch has ended
and outstanding translations == 0
and outstanding memory requests == 0
```

### 3.7 Python configuration and C++ behavior

gem5 commonly divides its model into two layers:

- Python creates components, selects parameters, and connects ports.
- C++ implements what those components do while the simulation runs.

A correct C++ memory requester still cannot work if its port is not connected
to a cache or interconnect in Python. Conversely, a correct port connection
does not fix incorrect C++ request behavior.

## 4. The main gem5 objects in a translated load

The following table maps each idea to its role in gem5.

| Idea | gem5 abstraction | Purpose |
| --- | --- | --- |
| Hardware component | `SimObject` | Holds modeled state and behavior |
| Executing hardware thread | `ThreadContext` | Provides registers, PC, process, CPU, and MMU context |
| Desired memory operation | `Request` | Stores virtual/physical address, size, flags, PC, and IDs |
| Address-translation unit | `BaseMMU` | Routes translation to the correct TLB |
| Translation cache | ISA-specific `TLB` | Looks up mappings and checks permissions |
| TLB-miss engine | Page-table walker | Reads page-table entries in full-system mode |
| Asynchronous completion | `BaseMMU::Translation` | Reports delay, success, or a fault |
| Architectural failure | `Fault` | Describes why the operation cannot proceed |
| Memory-system message | `Packet` | Carries the translated access through ports |
| Component connection | `Port` | Transfers requests, responses, and retries |

### 4.1 `Request` versus `Packet`

This distinction is essential.

A `Request` describes the original operation. It can contain:

- a virtual address;
- a physical address after translation;
- the access size;
- read/write-related flags;
- a requestor ID;
- the instruction address that caused the access; and
- a thread context ID.

A `Packet` is a message sent through the memory system. It references a
`Request`, adds a command such as `ReadReq`, and may carry data and sender state.

The usual ordering for a physically addressed cache is:

```text
Create Request with virtual address
                |
                v
Translate Request and set physical address
                |
                v
Create Packet from translated Request
                |
                v
Send Packet to cache
```

Creating or sending the ordinary cache packet before successful translation is
a correctness error.

### 4.2 Ports and retry

Ports connect memory-system components. A request port sends requests and
receives responses. A response port receives requests and sends responses.

Timing requests may encounter backpressure. A downstream component can reject
a request temporarily, after which the sender must wait for a retry signal.
`QueuedRequestPort` and `ReqPacketQueue` provide queueing and retry machinery so
the requesting component does not have to reimplement it.

Address translation and port retry are different delays:

```text
Translation delay: waiting to obtain a physical address
Port delay:        waiting for the memory system to accept a packet
Memory delay:      waiting for the accepted packet's response
```

## 5. A translated read in plain English

Before reading source code, be able to tell this story:

1. A simulated instruction calculates a virtual address.
2. The CPU creates a `Request` describing a read from that virtual address.
3. The CPU asks its MMU to perform a data translation.
4. The MMU selects the data TLB.
5. The TLB checks whether it already has a matching translation.
6. On a hit, the TLB checks permissions and writes the physical address into
   the request.
7. On a miss, gem5 either consults an emulated process page table or starts a
   modeled hardware page-table walk, depending on simulation mode.
8. Translation calls `finish()` with either `NoFault` or a fault.
9. On success, the requester creates a read packet from the translated request.
10. The packet travels through the cache and memory hierarchy.
11. A response returns containing the requested data.
12. The requester consumes the data and completes the operation.

Steps 1 through 8 are address generation and translation. Steps 9 through 12
are the memory-system access.

## 6. Functional, atomic, and timing translation

gem5 exposes different access styles for different modeling needs.

### 6.1 Functional

Functional operations answer the question "what is there?" without modeling
normal timing behavior. They are useful for activities such as loading a
program, debugger access, or inspecting simulator state.

`translateFunctional()` obtains a result synchronously. Using it in a modeled
accelerator load can produce correct bytes while skipping timing TLB and walker
behavior.

### 6.2 Atomic

Atomic operations complete synchronously but may return an aggregate latency.
They are useful in faster, less detailed simulation modes.

### 6.3 Timing

Timing operations represent asynchronous hardware interactions. A timing
translation takes a `BaseMMU::Translation` callback rather than returning the
eventual result directly.

```text
translateTiming(req, tc, callback, mode)
        |
        +-- immediate completion --> callback.finish(...)
        |
        `-- delayed completion ----> callback.markDelayed()
                                      ... later ...
                                      callback.finish(...)
```

Changing a component from functional to timing translation is therefore not
just an API substitution. It requires persistent state, callback ownership,
fault handling, counters, and a completion policy.

## 7. Syscall-emulation mode versus full-system mode

gem5 can run the repository's x86 workload in two importantly different modes.

### 7.1 Syscall-emulation mode

In syscall-emulation, or SE, mode, gem5 runs a user program without simulating
an entire guest operating system. gem5 represents the process and its mappings
using an `EmulationPageTable`.

On an x86 timing TLB miss in SE mode, the TLB looks up the virtual address in
that emulated page table. This lookup is performed inside the translation call,
so `finish()` can be called immediately even though the TLB recorded a miss.

Consequences:

- SE mode can test whether the virtual address maps to the right physical page.
- SE mode can exercise TLB hit and miss bookkeeping.
- A miss does not generate realistic hardware page-table-walker traffic.
- `markDelayed()` will not normally demonstrate a hardware page walk.

The current AMX testbench, `configs/amx/tb.py`, creates a `SimpleBoard` and calls
`set_se_binary_workload()`, so it is an SE-mode test.

### 7.2 Full-system mode

In full-system, or FS, mode, gem5 simulates a machine running a guest operating
system. The guest owns page tables in simulated memory.

On an x86 timing TLB miss in FS mode, the TLB starts its page-table walker. The
walker issues memory requests for page-table entries. Translation is marked
delayed, and `finish()` is called after the walk completes or faults.

Consequences:

- FS mode is required to validate real timing walker traffic.
- The walker port must be connected to the memory system.
- Page-table reads contend for modeled memory resources.
- The translation callback and its state must survive for later completion.

This difference is why an implementation can appear correct in the current SE
test while still containing a delayed-callback lifetime bug.

## 8. Following translation through the gem5 source

This section gives a deliberate reading order. Follow one function at a time;
do not try to understand the entire MMU at once.

### 8.1 Start at the generic MMU contract

Read:

- `src/arch/generic/mmu.hh`
- `src/arch/generic/mmu.cc`

Focus on:

- `BaseMMU::Mode`: `Read`, `Write`, or `Execute`;
- `BaseMMU::Translation`;
- `translateAtomic()`;
- `translateTiming()`; and
- `translateFunctional()`.

The generic `BaseMMU::translateTiming()` selects the instruction TLB for
`Execute` and the data TLB for `Read` or `Write`, then forwards the call.

### 8.2 Read a reference requester

Read:

- `src/cpu/simple/timing.cc`
- `src/cpu/translation.hh`

`TimingSimpleCPU::initiateMemRead()` creates a virtual request and a
`DataTranslation` callback. `DataTranslation::finish()` records translation
latency, tells the CPU that translation has finished, and deletes itself.

This is a useful reference because it shows the complete contract from the
requester's side:

```text
construct request and callback
        |
        v
call translateTiming
        |
        v
finishTranslation after callback
        |
        +-- fault --> return fault to instruction handling
        |
        `-- success --> send data packet
```

`src/cpu/minor/lsq.cc` provides another clear example. Its
`SingleDataRequest` implements the translation callback directly.

### 8.3 Follow the x86 TLB

Read `src/arch/x86/tlb.cc`, beginning with:

- `TLB::translate()`;
- `TLB::translateTiming()`; and
- `TLB::translateFunctional()`.

At a high level, `TLB::translate()`:

1. interprets the request flags and access mode;
2. applies x86 address and segment rules;
3. looks for a TLB entry when paging is enabled;
4. handles a miss using the FS walker or SE emulation page table;
5. checks page permissions;
6. combines the physical page with the virtual page offset;
7. calls `req->setPaddr(...)`; and
8. performs final physical-address handling.

`TLB::translateTiming()` then chooses between:

```text
not delayed: translation->finish(fault, req, tc, mode)
delayed:     translation->markDelayed()
```

The walker will call `finish()` later for the delayed case.

### 8.4 Follow the x86 walker

Read:

- `src/arch/x86/pagetable_walker.hh`
- `src/arch/x86/pagetable_walker.cc`
- `src/arch/x86/X86TLB.py`
- `src/arch/x86/X86MMU.py`

The walker is a state machine. Each state represents the level of the x86 page
table being processed. It creates packets for page-table entries, waits for
responses, updates permission information, and either descends to the next
level or completes the translation.

The Python files create the TLBs and walkers. The cache-hierarchy configuration
connects the walker ports to the memory system.

## 9. Following an AMX tile load in this repository

The AMX implementation is a helpful case study because one tile load becomes
many independently translated memory requests.

### 9.1 Entry from the workload

The test calls `amx_tile_loadd()` in files such as
`configs/amx/tests/tile_load_unaligned_test.cpp`. This is currently a gem5
pseudo-instruction, not a decoded architectural TILELOADD instruction.

The path is:

```text
amx_tile_loadd in workload
        |
        v
pseudo-instruction dispatch in src/sim/pseudo_inst.hh
        |
        v
pseudo_inst::amxLoadd in src/sim/pseudo_inst.cc
        |
        v
AmxAccl::queueAmxLoad
```

`queueAmxLoad()` records the `ThreadContext`, virtual base address, stride, tile
number, and a stable instruction ID in the accelerator's instruction queue.

### 9.2 Fragment generation

`AmxAccl::executeInstruction()` walks through the configured tile rows. Each
row starts at:

```text
row virtual address = base virtual address + row number * stride
```

A row can begin in the middle of a cache line, so the implementation divides it
at cache-line boundaries. Each fragment remembers two different offsets:

- where useful data begins in the returned cache line; and
- where that data belongs in the destination tile row.

For example, a 16-byte row starting at address offset 60 spans two lines:

```text
First cache line:   use bytes 60..63  -> tile bytes 0..3
Second cache line:  use bytes 0..11   -> tile bytes 4..15
```

The implementation requests the complete aligned cache lines and later copies
only the useful bytes.

### 9.3 Starting translation

For each fragment, the accelerator creates a `Request` containing the aligned
virtual cache-line address. It increments `outstandingTranslations`, creates an
`AmxTranslation`, and calls:

```cpp
tc->getMMUPtr()->translateTiming(
    req, tc, translation, BaseMMU::Read);
```

The mode must be `Read` because a tile load reads application memory.

The count is incremented before the call because `finish()` may run inline.

### 9.4 Translation completion

`AmxTranslation::finish()` forwards the request and fault to
`AmxAccl::finishTranslation()` and deletes the one-shot callback.

`finishTranslation()`:

1. finds the accelerator instruction by stable ID;
2. consumes one outstanding translation;
3. records the first fault, if any;
4. refuses to send a packet after a translation fault;
5. verifies that a successful request has a physical address;
6. records translation latency;
7. creates the read packet and sender state;
8. increments the outstanding memory-request count; and
9. schedules the packet on the AMX request port.

The stable ID is important. A raw pointer into a changing instruction queue
could become invalid while callbacks and responses are outstanding.

### 9.5 Memory completion

The AMX request port is connected through a crossbar to the L1 data cache by
`src/amx/amxXbar/amx_private_l1_private_l2_cache_hierarchy.py`.

When the response returns, `handleMemResponse()` uses its sender state to find:

- the AMX instruction;
- destination tile and row;
- cache-line offset;
- tile-row offset; and
- number of useful bytes.

It copies the data into the tile, decrements `outstandingRequests`, and checks
whether the instruction can finish.

### 9.6 Why dispatch completion needs its own flag

In SE mode, every translation could complete inline. Without a dispatch flag,
the first callback could see both outstanding counters at zero and erase the
instruction while `executeInstruction()` is still trying to create more
fragments.

The completion condition therefore includes
`translationDispatchComplete`:

```text
dispatch loop has finished
and no translations are outstanding
and no memory requests are outstanding
```

This is an example of re-entrancy affecting a state machine even though gem5's
event loop itself is single-threaded.

## 10. What the current AMX path does not yet prove

Using `translateTiming()` is necessary for timing translation, but it is not by
itself proof of a complete architectural implementation.

### 10.1 The current test is SE mode

The existing test can exercise immediate callbacks and validate mapped address
translation. It cannot validate a delayed x86 hardware page walk. An FS test is
needed for that.

### 10.2 Translation faults panic

The AMX implementation records a translation fault and eventually calls
`panic()`. That prevents a bad physical request, but it does not deliver a
restartable fault to the simulated instruction.

A complete architectural implementation must define how the instruction is
stopped, faulted, and restarted after the operating system resolves the page.
That is difficult while the operation begins through a pseudo-instruction and
continues independently of normal CPU instruction completion.

### 10.3 Squashing and context lifetime

The default `BaseMMU::Translation::squashed()` returns false. A CPU translation
callback normally knows whether its instruction has been squashed. The AMX
callback currently does not model that relationship.

The design must also ensure that its saved `ThreadContext` remains valid for
the full operation.

### 10.4 Translation resources are not fully modeled

Submitting a timing translation uses the TLB's timing interface, but the AMX
component still needs an explicit performance model for questions such as:

- How many new translations can AMX submit per cycle?
- How many may remain outstanding?
- Does AMX contend with CPU loads for TLB lookup bandwidth?
- How many page walks can proceed concurrently?
- When is the AMX instruction allowed to retire?

These are timing-model decisions, not virtual-to-physical address calculations.

## 11. A correctness checklist for translation clients

Use this checklist whenever you review or implement a gem5 component that
starts address translation.

### Request construction

- Does the request contain the virtual address, correct size, and correct mode?
- Are requestor, context, and PC identifiers appropriate?
- Is a large access split wherever the selected CPU or memory path requires?
- Can any fragment cross a page or cache-line boundary unexpectedly?

### Before `translateTiming()`

- Is all callback-visible state initialized first?
- Are outstanding counters incremented first?
- Can an inline callback safely run immediately?
- Does the callback have a clearly defined owner?

### In `markDelayed()`

- Does the requester remain alive?
- Is delayed state recorded if later logic depends on it?
- Can the request be squashed while the walk is outstanding?

### In `finish()`

- Is the returned access mode the expected mode?
- Is the fault checked before reading the physical address?
- Does success imply `req->hasPaddr()`?
- Is a packet created only after successful translation?
- Is translation latency recorded at the correct point?
- Is the callback deleted exactly once?

### Memory request and response

- Is port backpressure handled without dropping or duplicating a packet?
- Is sender state sufficient to match an out-of-order response?
- Is the outstanding memory count incremented before the packet can respond?
- Who deletes the packet and sender state?

### Completion and faults

- Can completion occur during the dispatch loop?
- Does completion require all translation and memory work to finish?
- Can one fault race with successful callbacks already in flight?
- Are already-issued requests safely drained after a fault?
- Is the fault delivered architecturally, deliberately suppressed, or only
  reported as an explicit model limitation?

## 12. Experiments that demonstrate correctness

Printing a plausible tile is useful, but a strong test should calculate an
expected result and fail automatically when bytes differ.

| Experiment | What it checks |
| --- | --- |
| Aligned mapped buffer | Basic virtual-to-physical-to-cache path |
| Unaligned buffer | Cache-line fragmentation and copy offsets |
| Rows spanning 4 KiB pages | Independent translation of each page |
| Non-contiguous physical pages | No assumption of physical contiguity |
| Cold then warm translation | TLB miss followed by a hit |
| Several outstanding fragments | Counter and callback-order safety |
| Multiple AMX instructions | Stable IDs and independent completion |
| Port backpressure | Queue and retry correctness |
| Unmapped address | Fault produces no memory packet |
| Protected mapping | Access mode and permission checks |
| SE and FS versions | Inline lookup versus delayed page walk |

Useful evidence includes:

- workload checksums or byte-for-byte comparisons;
- `AMX`, `TLB`, `PageTableWalker`, and cache debug traces;
- DTB access and miss statistics;
- assertions on counters, IDs, and address validity; and
- a debug or sanitizer build that detects ownership errors.

For each experiment, write down the expected sequence before running it. A
trace is much easier to interpret when it is being compared against a concrete
prediction.

## 13. A reusable implementation pattern

A new translated timing requester generally needs this shape:

```text
1. Construct a Request containing a virtual address.
2. Allocate persistent per-translation state.
3. Increment the outstanding count.
4. Call translateTiming().
5. Be prepared for finish() to run immediately.
6. In finish(), consume the outstanding translation.
7. If faulted, do not create or send a normal memory packet.
8. On success, require a physical address.
9. Create a Packet and attach stable response metadata.
10. Increment the memory count before sending.
11. Send through a retry-capable timing port.
12. On response, consume the memory count and free packet state.
13. Complete only after dispatch and both counters reach zero.
14. Define squash, fault, restart, and context-lifetime behavior.
```

The precise classes will vary, but these invariants recur across CPUs,
accelerators, and other memory requesters.

## 14. Check your understanding

You should be able to answer these questions without looking at the code:

1. Why can a request have both a virtual and a physical address?
2. What is the difference between a TLB hit and a page-table walk?
3. Why are address translation and a cache access separate operations?
4. What is the difference between a `Request` and a `Packet`?
5. Why can `finish()` run before `translateTiming()` returns?
6. Why must a counter be incremented before starting translation?
7. What does `markDelayed()` mean?
8. Why does an x86 SE-mode TLB miss not validate walker timing?
9. Why must a packet not be sent after a translation fault?
10. Why is a stable instruction ID safer than a pointer into a queue?
11. What work must finish before an AMX tile load is complete?
12. Why is panic-on-fault not the same as architectural fault handling?

Then practice with the source:

1. Start at `AmxAccl::executeInstruction()` and identify where the request has
   only a virtual address.
2. Find the exact x86 line that assigns the physical address.
3. Find the paths that call `finish()` immediately and `markDelayed()`.
4. Follow an FS walker completion back to the translation callback.
5. Identify every allocation and deletion of AMX translation and packet state.
6. For one unaligned row, calculate every requested cache line and copy offset
   by hand, then compare your prediction with an AMX debug trace.

## 15. Further reading

- [gem5 memory-system documentation](https://www.gem5.org/documentation/general_docs/memory_system/)
- [gem5 BaseMMU class reference](https://doxygen.gem5.org/release/current/classgem5_1_1BaseMMU.html)
- [gem5 syscall-emulation versus full-system overview](https://www.gem5.org/documentation/learning_gem5/part1/simple_config/)
- [Intel 64 and IA-32 software developer manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- `src/amx/docs/amx-system-overview.md`
- `src/amx/docs/tile-load.md`
