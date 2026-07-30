# AMX simulator source guide

This directory contains the gem5 SimObject that models Intel AMX tile state,
instruction scheduling, and timed tile-memory operations. The implementation is
split by responsibility so that `amx_accl.cc` remains a short overview of the
simulator's behavior.

## How the pieces fit together

AMX pseudo-instructions enter the SimObject through the public `startAmx*`
methods. An instruction is then queued, checked for hazards, executed, and (for
memory operations) completed asynchronously after translation and cache
responses.

```text
pseudo-instruction
        |
        v
 amx_accl.cc       create and queue an instruction; drive the issue loop
        |
        v
 amx_scheduler.cc  select the oldest instruction that is safe to issue
        |
        v
 amx_execution.cc  perform opcode-specific work and update the scoreboard
        |
        +--------------------------+
        | memory operation         | non-memory operation
        v                          v
 amx_memory.cc              opcode-specific completion
 translate -> cache -> copy
        |
        v
 amx_execution.cc  commit results, remove the instruction, and issue again
```

`amx_instruction.*` defines what an instruction contains and how dependencies
between instructions are detected. `amx_tile.*` owns the tile register and tile
configuration representations used by execution and memory handling.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `AmxAccl.py` | Declares the Python SimObject, its C++ class, memory-side port, and configurable parameters. |
| `SConscript` | Registers the SimObject, implementation files, and `AMX` debug flag with the gem5 build. |
| `amx_accl.hh` | Declares the `AmxAccl` interface and groups its private operations and state by subsystem. It is the shared contract between the implementation files. |
| `amx_accl.cc` | Contains only the architectural entry points, issue loop, and high-level opcode dispatch. Start here to understand overall behavior. |
| `amx_sim_object.cc` | Constructs the SimObject, connects it to the CPU, exposes its port, and handles memory-port callbacks. |
| `amx_instruction.hh` / `amx_instruction.cc` | Define queued instructions, their lifecycle and failure state, factory functions, and tile read/write dependency checks. |
| `amx_scheduler.cc` | Searches the instruction queue, applies configuration barriers, checks active and older-instruction hazards, and removes completed instructions. |
| `amx_execution.cc` | Implements tile-load and tile-configuration behavior, completion, configuration commit, scoreboard updates, and the compute/store placeholders. |
| `amx_memory.cc` | Splits reads at cache-line boundaries, performs address translation, sends timing requests, routes response bytes, and detects memory-operation completion. |
| `amx_tile.hh` / `amx_tile.cc` | Define tile/configuration data, decode and validate configurations, clear tile state, and format tile contents for debug traces. |
| `amxXbar/` | Contains the Python cache-hierarchy configuration used to connect AMX memory traffic. It is separate from the C++ execution model. |
| `notes/` | Contains design research and implementation notes; it is not compiled into the simulator. |

## Instruction lifecycles

### Tile load

1. `startAmxLoad()` creates an instruction and calls `tryIssue()`.
2. `findReadyInstruction()` rejects it while an older or active instruction has
   a conflicting tile dependency.
3. `executeLoadInstruction()` marks the destination tile busy, clears it, and
   dispatches one logical read for every configured row.
4. `amx_memory.cc` splits each row across cache lines when necessary. Every
   chunk is translated and sent through the timing memory port.
5. Responses are copied into the correct tile row. When translation dispatch,
   translations, and requests have all finished, `finishLoadInstruction()`
   releases the scoreboard entry, traces the tile, removes the instruction, and
   runs the issue loop again.

### Tile configuration

1. `startAmxLoadConfig()` queues a configuration-memory read.
2. The scheduler treats configuration as a full barrier: it must reach the
   queue front and all tile activity must be idle, while younger instructions
   cannot pass it.
3. The 64-byte configuration is read through the same translation and timing
   memory path as a tile load.
4. `finishConfigInstruction()` decodes and validates the bytes after memory and
   minimum configuration latency have completed.
5. `commitTileConfig()` installs the new layout and clears all tile data.

## Important invariants

- Configuration instructions are full ordering barriers.
- The tile scoreboard represents currently executing work. Queue-order hazards
  against older instructions are checked separately by the scheduler.
- Memory completion is allowed only after dispatch is complete and both the
  outstanding-translation and outstanding-request counters reach zero. This is
  necessary because translation callbacks can occur immediately.
- Asynchronous callbacks carry an instruction ID and look the instruction up in
  the queue instead of retaining a queue-entry pointer across events.
- Tile contents are cleared whenever a configuration is committed and before a
  tile load begins writing its destination.
- Tile formatting belongs in `amx_tile.cc`; execution code should only request
  a trace of a tile.

## Where new code should go

- Add or change architectural entry points and top-level dispatch in
  `amx_accl.cc`.
- Add instruction fields, factories, or dependency semantics in
  `amx_instruction.*`.
- Add readiness and ordering rules in `amx_scheduler.cc`.
- Add opcode behavior and completion in `amx_execution.cc`.
- Add translation, packet, cache-line, or response behavior in
  `amx_memory.cc`.
- Add tile layout, configuration validation, or tile display helpers in
  `amx_tile.*`.
- Add lifecycle, CPU connection, or port setup in `amx_sim_object.cc`.
- Add user-configurable SimObject parameters in `AmxAccl.py`.

When adding an opcode, its instruction representation, dependency behavior,
execution path, completion path, and scoreboard effects should be considered
together. Compute and store currently have scheduling placeholders but do not
yet implement functional completion.

Use gem5's `AMX` debug flag to see queueing, memory, configuration, and tile
trace messages from this component.
