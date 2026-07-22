# AMX Accelerator System Overview

This document describes the current gem5 AMX accelerator implementation. It
also distinguishes the code that works today from the intended memory-system
design that still needs to be connected.

## High-Level System

```text
User program
    |
    | amx_tile_loadd(tile, address, stride)
    v
gem5 pseudo-instruction handler
src/sim/pseudo_inst.cc
    |
    | cpu->getAmxAccl()
    | accelerator->startAmxLoad(...)
    v
+----------------------------------------------------------+
|                     AMX Accelerator                      |
|                                                          |
|  +----------------------+                                |
|  | Instruction Queue    |                                |
|  |                      |                                |
|  | LOAD TMM0            |                                |
|  | LOAD TMM1            |                                |
|  | COMPUTE ...          |                                |
|  +----------+-----------+                                |
|             |                                            |
|             v                                            |
|  +----------------------+     +-----------------------+   |
|  | Dependency Scheduler |<--->| Tile Scoreboard       |   |
|  | findReadyInstruction |     | readers / writers     |   |
|  +----------+-----------+     +-----------------------+   |
|             |                                            |
|             v                                            |
|  +----------------------+                                |
|  | Instruction Executor |                                |
|  | executeInstruction   |                                |
|  +----------+-----------+                                |
|             |                                            |
|     Tile load is split into cache-line requests          |
|             |                                            |
|             v                                            |
|  +----------------------+                                |
|  | AmxRequestPort       |                                |
|  | QueuedRequestPort    |                                |
|  |                      |                                |
|  | - request queue      |                                |
|  | - retry handling     |                                |
|  | - response callback  |                                |
|  +----------+-----------+                                |
|             |                                            |
|             | responses                                  |
|             v                                            |
|  +----------------------+     +-----------------------+   |
|  | Response Handler     |---->| Tile Register File    |   |
|  | handleMemResponse    |     | TMM0 ... TMM7         |   |
|  +----------------------+     | 16 rows x 64 bytes    |   |
|                               +-----------------------+   |
+-------------+--------------------------------------------+
              |
              | mem_side port
              v
      Intended arbitration point
              |
        +-----+-----+
        | Coherent  |
CPU --->| crossbar  |<--- AMX
        +-----+-----+
              |
              v
           L1 D-cache
              |
              v
        L2 / memory system
```

The last portion is the intended design. The current test configuration creates
an `AmxAccl`, but does not yet connect `mem_side` to a crossbar/cache port or
call `setCPU()`. Therefore, the memory-system path is not operational yet.

## Main Components

### Tile Register File

The accelerator contains eight modeled AMX tile registers:

```cpp
TileReg tiles[NUM_TILES];
```

Conceptually:

```text
tiles
|-- TMM0: 16 rows x 64 bytes
|-- TMM1: 16 rows x 64 bytes
|-- TMM2: 16 rows x 64 bytes
|-- ...
`-- TMM7: 16 rows x 64 bytes
```

Each register has storage for the maximum tile size:

```cpp
struct TileReg
{
    uint16_t rows;
    uint16_t colbytes;
    int8_t data[16][64];
};
```

`currentCfg` determines how many rows and bytes are active for each tile.

### Instruction Queue

When the accelerator receives a tile-load operation, it creates an internal
instruction rather than completing the load immediately:

```text
AmxInst
|-- unique instruction ID
|-- operation: load / compute / store
|-- destination tile
|-- source tiles
|-- base memory address
|-- stride
|-- outstanding request count
|-- state
`-- failure information
```

The instruction is placed in:

```cpp
std::deque<AmxInst> instructionQueue;
```

Its state moves through:

```text
PENDING ---> EXECUTING ---> COMPLETED
```

### Scoreboard

The scoreboard records whether each tile is currently being read or written:

```cpp
struct ScoreBoardEntry
{
    int readerCount;
    bool writeActive;
};
```

For example:

```text
LOAD TMM0
COMPUTE TMM1, TMM0, TMM2
              ^
              cannot read TMM0 until the load finishes
```

The scheduler checks for:

- RAW: read after write
- WAW: write after write
- WAR: write after read

It checks both the active scoreboard and earlier instructions in the queue.

## Tile-Load Control Flow

```text
startAmxLoad()
      |
      v
Create AmxInst
      |
      v
Push into instructionQueue
      |
      v
tryIssue()
      |
      v
findReadyInstruction()
      |
      +--- dependency exists ---> remain PENDING
      |
      +--- no dependency -------> executeInstruction()
                                      |
                                      v
                              Mark tile write-active
                                      |
                                      v
                              Split tile into rows
                                      |
                                      v
                         Split rows at cache-line boundaries
                                      |
                                      v
                           Translate virtual addresses
                                      |
                                      v
                         Schedule memory request packets
                                      |
                                      v
                         Wait for all packet responses
                                      |
                                      v
                         finishLoadInstruction()
```

## Pseudocode

### Entering the Accelerator

```text
function amxLoadd(thread, destinationTile, sourceAddress, stride):
    cpu = thread.cpu
    accelerator = cpu.amxAccelerator

    if accelerator exists:
        accelerator.startAmxLoad(
            thread,
            destinationTile,
            sourceAddress,
            stride
        )
    else:
        warn that no accelerator is attached
```

### Adding a Load to the Queue

```text
function startAmxLoad(thread, tile, address, stride):
    verify thread is valid
    verify tile is between 0 and 7

    instruction = new AMX instruction
    instruction.id = next unique ID
    instruction.opcode = LOAD
    instruction.destination = tile
    instruction.address = address
    instruction.stride = stride
    instruction.state = PENDING
    instruction.outstandingRequests = 0

    append instruction to instructionQueue

    tryIssue()
```

The unique ID is needed because memory responses may return out of order.

### Selecting an Instruction

```text
function tryIssue():
    if instructionQueue is empty:
        return

    instruction = findReadyInstruction()

    if an instruction was found:
        executeInstruction(instruction)
```

The scheduler does not necessarily select the instruction at the front of the
queue. It scans the queue for the first instruction that can safely execute.

```text
function findReadyInstruction():
    for each pending instruction in queue:
        check active tile readers and writers
        check dependencies against earlier instructions

        if no hazard:
            return pointer to instruction

    return no instruction
```

## Splitting a Tile Load into Requests

For a tile with 16 rows, 64 bytes per row, and a stride of 64 bytes, an aligned
load creates 16 cache-line requests:

```text
Row 0  -> one 64-byte request
Row 1  -> one 64-byte request
...
Row 15 -> one 64-byte request
```

An unaligned row may cross a cache-line boundary:

```text
Cache line A                    Cache line B
+-----------------------------+-----------------------------+
| unused | first 48 row bytes | remaining 16 bytes | unused |
+-----------------------------+-----------------------------+
         ^
         row begins here
```

That row becomes two requests:

```text
Request 1: read cache line A and copy 48 bytes
Request 2: read cache line B and copy 16 bytes
```

The request-generation logic is:

```text
function executeLoad(instruction):
    mark destination tile as being written
    instruction.state = EXECUTING
    instruction.outstandingRequests = 0

    for each active tile row:
        currentAddress = baseAddress + row * stride
        bytesRemaining = configured row size
        tileRowOffset = 0

        while bytesRemaining > 0:
            cacheLineAddress = align currentAddress down to 64 bytes
            cacheOffset = currentAddress mod 64

            availableBytes = 64 - cacheOffset
            chunkSize = min(bytesRemaining, availableBytes)

            translate cacheLineAddress

            if translation fails:
                record translation failure
                stop creating any more requests
                break out of all row processing

            create 64-byte read packet

            attach sender state:
                instruction ID
                destination tile
                tile row
                cache-line offset
                tile-row destination offset
                number of bytes to copy

            increment instruction.outstandingRequests
            schedule packet through queued request port

            currentAddress += chunkSize
            tileRowOffset += chunkSize
            bytesRemaining -= chunkSize

    if no requests were generated:
        finish the instruction immediately
```

## Sender-State Bookkeeping

Responses may arrive in a different order from the requests:

```text
Sent order:      row 0, row 1, row 2, row 3
Response order:  row 2, row 0, row 3, row 1
```

The response packet therefore carries an `AmxSenderState`:

```text
Response packet
`-- AmxSenderState
    |-- instruction ID
    |-- destination tile
    |-- row number
    |-- offset within returned cache line
    |-- offset within destination tile row
    `-- bytes to copy
```

Here, bookkeeping means the metadata used to match an asynchronous response to
the instruction, tile, and row that requested it.

## Handling a Memory Response

```text
function handleMemResponse(packet):
    retrieve sender state from packet

    use senderState.instructionID
        to find the owning instruction

    if packet reports an error:
        record memory-response failure

    else if packet has no data:
        record missing-data failure

    else:
        validate tile, row, and offsets

        source = packet.data + senderState.cacheOffset
        destination =
            tile[tileNumber][rowNumber] + senderState.rowOffset

        copy senderState.bytesToCopy bytes
            from source to destination

    decrement instruction.outstandingRequests

    delete sender state
    delete packet

    if outstandingRequests == 0:
        finishLoadInstruction(instruction)
```

## Detecting Load Completion

The outstanding-request counter represents:

```text
outstandingRequests =
    number of packets scheduled - number of responses processed
```

For example:

```text
Schedule request A: 0 -> 1
Schedule request B: 1 -> 2
Schedule request C: 2 -> 3

Response B:          3 -> 2
Response A:          2 -> 1
Response C:          1 -> 0  => load complete
```

The response order does not matter. Completion occurs only after every
scheduled request has produced a response.

```text
function finishLoadInstruction(instruction):
    verify outstandingRequests == 0

    mark instruction COMPLETED
    release destination tile's scoreboard write lock

    if translation failed:
        report unsupported asynchronous translation fault

    if a memory response failed:
        report memory failure

    if a response contained no data:
        report missing-data failure

    otherwise:
        tile load succeeded
        optionally print tile contents
        remove instruction from queue
        try to issue another instruction
```

Releasing the scoreboard allows dependent instructions to execute:

```text
Before completion:
    TMM0.writeActive = true
    dependent instructions cannot use TMM0

After completion:
    TMM0.writeActive = false
    dependent instructions may issue
```

## Queued Request Port and Retries

The accelerator schedules each packet with:

```cpp
memSidePort.schedTimingReq(pkt, curTick());
```

The queued port conceptually performs:

```text
function scheduleRequest(packet):
    put packet into request queue
    attempt to send it

    if the downstream cache rejects it:
        keep packet queued
        wait for recvReqRetry()

    when the retry notification arrives:
        attempt to send the same packet again
```

A temporarily rejected timing request is therefore retained by the queued
port. The AMX accelerator does not need a second, separate retry queue.

## Current and Unfinished Work

| Area | Current state |
|---|---|
| AMX SimObject | Implemented |
| Per-CPU accelerator parameter | Implemented |
| Pseudo-instruction entry point | Implemented |
| Internal instruction queue | Implemented |
| Basic dependency scheduler | Implemented |
| Tile-load request splitting | Implemented |
| Unaligned/cache-line splitting | Implemented |
| Request retry queue | Implemented through `QueuedRequestPort` |
| Out-of-order response matching | Implemented |
| Load completion tracking | Implemented |
| Compute behavior | Placeholder |
| Store behavior | Placeholder |
| Timing address translation | Not implemented; currently functional |
| Architectural fault delivery | Not implemented |
| Configurable issue limits | Declared but not connected |
| AMX port connection | Not configured yet |
| Shared CPU/AMX cache arbitration | Not configured yet |
| Accurate CPU load-port contention | Not modeled yet |

The most important architectural gap is the memory connection. The AMX port
exists, but it still needs to be connected to an arbitration point that shares
the CPU's L1 data-cache path. Until that is done, the model has the
request-generation machinery but does not model CPU-versus-AMX contention.
