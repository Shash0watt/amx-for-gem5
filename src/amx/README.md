# AMX Simulator Source Guide

This directory has the gem5 SimObject impleentation that models the intel AMX acceletor. 
At a high level you can view this as an extension to the execute state of the CPU pipeline.

## Current Implementation Notes 
gem5 decodes the instruction and passes it here for us to execute. 
Right now we use the pseduo-insts, to implement this functionality. Eventually the goal is to have the instructions handled completley by the CPU relying on this SimObject only for the tile state, latencies & throughput of the accelerator. 

The implementation is split by responsibility so that `amx_accl.cc` remains a short overview of the simulator's behavior.

**Use the`AMX` debug flag in  gem5 to see queueing, memory, configuration, and tile
trace messages from this component.**

## How the pieces fit together

AMX pseudo-instructions enter the SimObject through the public `queueAmx*`
methods. An instruction is queued, checked for dependencies and pipeline
availability, executed, and completed after its modeled latency. Memory
operations must also wait for address translation and cache responses.

```text
pseudo-instruction
        |
        v
 amx_accl.cc       create and queue an instruction; drive the issue loop
        |
        v
 amx_scheduler.cc  find safe work and check pipeline availability
        |
        v
 amx_accl.cc       reserve the pipeline
        |
        +--------------------------+
        |                          |
        v                          v
 amx_execution.cc           amx_latency.cc
 start opcode work          wait for fixed latency
        |
        v
 amx_memory.cc              (loads and stores)
 translate/cache, copy data
        |                          |
        +------------+-------------+
                     |
 (wait for both memory & latency to finish)
                     |
                     v
 amx_execution.cc  finish the operation, release its tile/resource,
                   remove it from the queue, and issue again
```

The memory and latency branches overlap. Tile loads, stores, and configuration
reads start both branches when they issue and join them in the unified
`completeInstructionIfReady()` path. Finishing only one branch is not enough
to complete an operation. Dot product and zero use only the fixed-latency
branch because they do not access memory.

## File Naming
`instruction_amx.*` defines what an instruction contains and how dependencies
between instructions are detected. `resource_amx.*` models how frequently each
pipeline can accept work. `tile_amx.*` owns the tile registers and tile
configuration used by execution and memory handling.

The filenames make this split visible: `amx_*.cc` files drive the control flow,
while `*_amx.hh` and `*_amx.cc` files provide supporting types and helper
functions.

## Scheduling and delayed events

`findReadyInstruction()` scans the queue from oldest to youngest and returns
the first instruction that is safe to issue now.

```text
findReadyInstruction()
        |
        | Is the instruction safe?
        | - no tile dependency
        | - no configuration barrier
        v
 Is its pipeline available now?
        |
   +----+----+
   |         |
  No        Yes
   |         |
   v         v
Schedule     Return instruction
issue retry  to tryIssue()
   |         |
   |         v
   |   executeInstruction()
   |         |
   |         +--> reserve pipeline
   |         +--> record issue time
   |         +--> schedule latency deadline
   |         +--> begin opcode-specific work
   |
   v
issueRetryEvent -> tryIssue() -> scan again
```

The main scheduling rules are:

- Configuration is a full barrier. It must be at the front of the queue with
  every tile idle, and younger work cannot pass it.
- Active tile readers and writers are tracked by the scoreboard.
- Dependencies on older instructions are also checked directly in the queue.
  This catches older instructions that have not reached the scoreboard yet.
- An unavailable pipeline blocks only instructions using that pipeline. The
  scheduler continues looking for independent work on other pipelines.
- Executing instructions remain in the queue until completion, but the
  scheduler can skip over them to find independent pending work.
- `tryIssue()` keeps issuing ready work in the current tick until the scheduler
  cannot find anything else to start.

Three kinds of delayed activity wake the model later:

| Activity | Why it is scheduled | What happens when it runs |
| --- | --- | --- |
| `issueRetryEvent` | An otherwise-ready instruction is waiting for its pipeline's next issue cycle. | Calls `tryIssue()` to scan the queue again. |
| `latencyEvent` | An issued load, dot product, zero, store, or configuration has not reached its minimum execution latency. | Marks every instruction due at that tick as having finished its latency wait. |
| Memory callbacks | A translation or cache request is still outstanding. Gem5's MMU and memory system control this timing. | Drain memory counters and check whether the memory stage is complete. |

The retry event controls when work may **start**. The latency event and memory
callbacks control when issued work may **finish**. An operation with memory
access requires both its latency and memory paths to finish; whichever path
finishes last completes the instruction.

## Instruction lifecycles

### Tile load

1. `queueAmxLoad()` creates an instruction and calls `tryIssue()`.
2. `findReadyInstruction()` rejects it while an older or active instruction has
   a conflicting tile dependency.
3. `executeLoadInstruction()` marks the destination tile busy, clears it, and
   dispatches one logical read for every configured row.
4. `amx_memory.cc` splits each row across cache lines when necessary. Every
   chunk is translated and sent through the timing memory port.
5. Responses are copied into the correct tile row. The load finishes only when
   both memory and its configured minimum latency have completed.
6. `finalizeInstruction()` releases the scoreboard and load-pipeline state,
   removes the instruction, and runs the issue loop again.

### Tile configuration

1. `queueAmxLoadConfig()` queues a configuration-memory read.
2. The scheduler treats configuration as a full barrier: it must reach the
   queue front and all tile activity must be idle, while younger instructions
   cannot pass it.
3. The 64-byte configuration is read through the same translation and timing
   memory path as a tile load.
4. `finalizeInstruction()` decodes and validates the bytes after memory and
   minimum configuration latency have completed.
5. `commitTileConfig()` installs the new layout and clears all tile data.

### Tile store

1. `queueAmxStore()` creates a store whose source tile is a read dependency.
2. The scheduler waits for older writers to that tile, then the store reserves
   a tile-reader scoreboard entry and the store pipeline.
3. `executeStoreInstruction()` dispatches configured rows beginning at
   `startRow`, using the supplied base address and stride.
4. `amx_memory.cc` splits each row at cache-line boundaries, translates with
   write access, copies tile bytes into write packets, and waits for responses.
5. The store completes only after both all memory activity and its configured
   minimum latency finish. Successful completion clears `startRow`, releases
   the tile reader and store resource, removes the instruction, and retries
   queued work.

A configuration queued after a store is a useful completion barrier: it cannot
commit while the store still holds its source-tile reader reservation. Tests
that need to inspect stored bytes can put a verification configuration and
tile reload after that barrier instead of assuming pseudo-op calls synchronize
the guest CPU with the accelerator.

### BF16 dot product

A dot product moves from submission to completion through this path:

```text
Guest pseudo-instruction
        |
        v
queueAmxDotProduct()
        |
        v
Create a Pending instruction and add it to instructionQueue
        |
        v
findReadyInstruction()
        |
        +--> wait for an older configuration barrier
        +--> wait for tile dependencies
        +--> wait for the dot-product pipeline
        |
        v
executeInstruction()
        |
        +--> reserve the dot-product pipeline
        +--> record issueTick
        +--> schedule the latency deadline
        |
        v
executeDotProductInstruction()
        |
        +--> validate the tile shapes
        +--> change state to Executing
        +--> reserve its tile reads and write
        |
        v
gem5 latencyEvent
        |
        v
instructionLatencyElapsed()
        |
        v
finalizeInstruction()
        |
        +--> doDotProductBF16()
        +--> update the destination tile
        +--> release tile and pipeline state
        +--> remove the instruction from the queue
        |
        v
tryIssue() remaining work
```

1. **Submission:** The pseudo-instruction calls `queueAmxDotProduct()` with a
   destination and two source tiles. The function validates the tile numbers,
   assigns a unique ID, creates the instruction, adds it to the queue, and
   calls `tryIssue()`.

2. **Waiting in the scheduler:** `findReadyInstruction()` checks configuration
   barriers, dependencies on older work, active tile readers and writers, and
   dot-product pipeline availability. Independent younger work may issue while
   this instruction is blocked.

3. **Issue and timing:** `executeInstruction()` reserves the dot-product
   resource, records the issue time, and schedules a deadline using
   `dotProductLatency`. The resource's initiation interval determines when a
   later independent dot product may issue; it does not have to wait for this
   instruction to complete.

4. **Tile reservation:** `executeDotProductInstruction()` calls
   `validateDotProductOp()` to verify that the configured tile shapes are
   compatible. It then marks the instruction as executing and reserves its
   tile accesses:

   ```text
   destination -> reader and writer
   source 1    -> reader
   source 2    -> reader
   ```

   The destination is also read because the operation accumulates into its
   existing FP32 values. These reservations prevent conflicting operations
   from using the tiles before the result is ready.

5. **Latency event:** When gem5 reaches the deadline,
   `processLatencyEvent()` calls `instructionLatencyElapsed()`. That marks the
   latency as elapsed and sends the dot product to
   `finishDotProductInstruction()`.

6. **Calculation and completion:** `doDotProductBF16()` reads BF16 pairs from
   the source tiles, multiplies them, and accumulates FP32 results into the
   destination. Completion then releases the tile reservations, records that
   the resource is no longer in flight, removes the instruction from the
   queue, and calls `tryIssue()` so newly unblocked work can start.

## Timing model

Latency and issue throughput describe different things:

- **Latency** is how long one operation takes to produce its result.
- **Issue throughput** is the minimum time before the same pipeline can accept
  another operation. (defined by intel)

For example, a dot product may still be in flight when the pipeline becomes
ready to accept a later independent dot product. If one pipeline is
unavailable, the scheduler continues looking for independent work that uses a
different pipeline. `AmxAccl.py` contains the default timing values.

## Important invariants

- Configuration instructions are full ordering barriers.
- The tile scoreboard represents currently executing work. Queue-order hazards
  against older instructions are checked separately by the scheduler.
- Pipeline issue restrictions do not block independent operations that use a
  different resource.
- Loads complete only after both their memory work and fixed latency finish.
- Stores complete only after both their write responses and fixed latency
  finish, and write exactly the configured bytes rather than whole cache lines.
- Memory completion is allowed only after dispatch is complete and both the
  outstanding-translation and outstanding-request counters reach zero. This is
  necessary because translation callbacks can occur immediately.
- Asynchronous callbacks carry an instruction ID and look the instruction up in
  the queue instead of retaining a queue-entry pointer across events.
- Tile contents are cleared whenever a configuration is committed and before a
  tile load begins writing its destination.
- Tile formatting belongs in `tile_amx.cc`; execution code should only request
  a trace of a tile.

## Where new code should go

- Add or change architectural entry points and top-level dispatch in
  `amx_accl.cc`.
- Add instruction fields, factories, or dependency semantics in
  `instruction_amx.*`.
- Add readiness and ordering rules in `amx_scheduler.cc`.
- Add pipeline throughput behavior in `resource_amx.*`.
- Add fixed-latency event behavior in `amx_latency.cc`.
- Add opcode behavior and completion in `amx_execution.cc`.
- Add BF16 dot-product validation or math in `dp_math_amx.hh`.
- Add translation, packet, cache-line, or response behavior in
  `amx_memory.cc`.
- Add tile layout, configuration validation, or tile display helpers in
  `tile_amx.*`.
- Add lifecycle, CPU connection, or port setup in `amx_sim_object.cc`.
- Add user-configurable SimObject parameters in `AmxAccl.py`.


