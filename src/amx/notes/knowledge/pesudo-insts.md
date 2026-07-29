# gem5 Pseudo-Instructions: Execution Mechanics, Flags, & Quirks

This document summarizes the architecture, pipeline behavior, static instruction flags, and quirks associated with **pseudo-instructions** (`m5ops`) in gem5, with a particular focus on how out-of-order CPU models (such as `O3CPU`) process non-speculative operations safely.

---
## 1. Overview of gem5 Pseudo-Instructions (`m5ops`)

gem5 pseudo-instructions (commonly referred to as `m5ops`) are special instructions embedded into target workload binaries (e.g., `m5_exit`, `m5_checkpoint`, `m5_reset_stats`, `m5_switch_cpu`, `m5_work_begin`, `m5_work_end`). 

Instead of performing standard architectural hardware operations, they serve as a **communication channel between the workload executing inside the simulator and the gem5 simulation host**.

---

## 2. The `IsNonSpeculative` Flag

### Purpose
In out-of-order CPUs, instructions normally issue as soon as their source operands become available in the Instruction Queue, regardless of whether older branches have resolved.

However, pseudo-instructions carry **irreversible host-side side effects** (such as stopping simulation, writing disk files, or altering simulator state).

### Why Speculative Execution is Dangerous
If a pseudo-instruction were executed speculatively:
* A branch predictor misprediction could fetch an `m5_exit()` instruction down an invalid execution path.
* If executed speculatively, gem5 would **immediately terminate the entire simulation session prematurely**, even though that branch path was architecturally invalid.
* Similarly, an `m5_checkpoint()` executed speculatively would write out a full simulation checkpoint file for an instruction sequence that never actually occurred.

---

## 3. How gem5’s O3 CPU Enforces `IsNonSpeculative`

The O3 CPU pipeline handles `IsNonSpeculative` instructions through strict stalling mechanisms:

```
[ Rename ] ──► [ IEW Stage ] ───────────────────────► [ ROB (Commit) Head ]
                   │                                          │
                   ├── Identifies isNonSpeculative()           ├── Reaches head of ROB
                   ├── SKIPS normal Out-of-Order issue        ├── Proves instruction is 100% non-speculative
                   └── Inserts into insertNonSpec()           └── Signals IEW: "Safe to Execute Now!"
```

### Step-by-Step Pipeline Mechanics:

1. **IEW Interception (`src/cpu/o3/iew.cc`)**:
   When an instruction enters the IEW stage, gem5 checks `inst->isNonSpeculative()`. Instead of adding it to the normal Instruction Queue (`IQ`) for out-of-order dispatch, it intercepts the instruction:
   ```cpp
   if (add_to_iq && inst->isNonSpeculative()) {
       inst->setCanCommit();
       instQueue.insertNonSpec(inst); // Queued separately as non-speculative
       add_to_iq = false;             // Do NOT dispatch to execution unit yet!
   }
   ```

2. **ROB Queueing (`src/cpu/o3/rob.cc`)**:
   The instruction remains unexecuted and moves into the Reorder Buffer (`ROB`) in sequence number order.

3. **Commit Stage Verification (`src/cpu/o3/commit.cc`)**:
   The instruction stays idle until it reaches the **head of the ROB**.
   Reaching the head of the ROB proves that **all older instructions have committed**, all prior branches were correctly predicted, and no speculative path remains above this instruction.

4. **Triggered Execution**:
   Once at the head of the ROB, Commit signals IEW (`toIEW->commitInfo[tid].nonSpecSeqNum`), giving permission for the pseudo-instruction to finally execute its simulation side-effect.

---

## 4. Other Static Instruction Flags for Pseudo-Insts

In `src/cpu/StaticInstFlags.py`, pseudo-instructions carry several related flags:

| Flag Name | Purpose & Effect on Pipeline |
| :--- | :--- |
| **`IsPseudo`** | Explicitly marks the instruction as a gem5 pseudo-op (`static_inst.hh`). Used by disassemblers (like Capstone) and debug tracers (`DPRINTF`) to format output correctly. |
| **`IsNonSpeculative`** | Prevents out-of-order execution before reaching the head of the ROB. |
| **`IsUnverifiable`** | Tells gem5's `CheckerCPU` runtime validation tool to **skip verifying this instruction** (`cpu/checker/cpu.hh`). Since pseudo-ops interact with simulator host state (reading wall-clock time, dumping host files), their results cannot be deterministically replayed or verified. |
| **`IsSerializeBefore`** | Forces the CPU to wait until **all older instructions in the ROB have committed** before this pseudo-op is allowed to execute. |
| **`IsSerializeAfter`** | Squashes or holds back all **younger instructions fetched after this pseudo-op** until the pseudo-op completely finishes. |
| **`IsQuiesce`** | Used by `m5_quiesce()` or `m5_quiesce_ns()`. Instructs the CPU scheduler to suspend the CPU thread context until an interrupt or timer event occurs. |

---

## 5. Architectural Quirks & Gotchas

### A. ISA-Specific Opcode Encodings
Pseudo-instructions use reserved/unused opcode space specific to each target ISA:
* **x86**: Uses a 2-byte unused opcode (`0x0F, 0x04`) followed by an immediate byte specifying the m5op function code.
* **ARM (AArch64)**: Uses special reserved instruction encodings (`0xD5000000` family / custom coprocessor space).
* **RISC-V**: Uses custom system opcode space (`0x00000073` variants).

### B. Register ABI & Compiler Optimization Hazards
Pseudo-ops receive arguments through the target ISA's standard integer argument registers (e.g., `x0`–`x5` on ARM64, `rdi`/`rsi`/`rdx` on x86, `a0`–`a5` on RISC-V).

> **Warning**: When writing custom inline assembly for m5ops in C/C++, compiler optimizations can reorder or eliminate register writes if inline assembly clobber lists (`"memory"`, specific register registers) are not declared strictly. Always use gem5's provided `m5op.h` header macros.

### C. Pipeline Draining on CPU Switching (`m5_switch_cpu`)
Certain pseudo-ops (like `m5_switch_cpu` or ROI markers `m5_work_begin`/`m5_work_end`) trigger CPU model switching (e.g. booting in Atomic/KVM and switching to `O3CPU` for ROI):
* Executing `m5_switch_cpu` triggers a simulation event.
* The simulator initiates a **pipeline drain**: it halts fetching, lets in-flight instructions commit, flushes speculative queues, serializes thread state, and hands control over to the new CPU model.

### D. Full-System (FS) vs. Syscall Emulation (SE) Mode
* `m5_exit`: Exits the simulation in both SE and FS modes.
* `m5_checkpoint`: In FS mode, writes full CPU, memory, and device state. In SE mode, it writes out CPU and virtual memory space state.

### E. Address-Based Triggers (MMIO)
As an alternative to special instruction opcodes, gem5 supports **Address-Based Pseudo-ops**:
* Writing a specific command payload to a reserved physical address (e.g., `0xffff0000` MMIO region) triggers the simulator's pseudo-inst handler without executing a special opcode instruction. Useful when running inside real hardware hypervisors where custom opcodes cause invalid instruction traps.
