#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

// Every issued AMX instruction (except DumpState) must wait a minimum number
// of cycles before it may complete.

// 1. at issue, scheduleInstructionLatency() schedules one event per
//    instruction at issueTick + latency
// 2. when that event fires, instructionLatencyElapsed() marks the latency as
//    finished and checks whether the instruction is ready to retire
//
// The event carries the instruction id rather than a pointer. If the
// instruction has already left the queue (e.g. it failed and finished early)
// the lookup finds nothing and the event does nothing.

void
AmxAccl::scheduleInstructionLatency(uint64_t instruction_id, Cycles latency)
{
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction,
             "AMX scheduled latency for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));

    // Measure from the issue time, not from a later clock edge.
    const Tick deadline = instruction->issueTick + cyclesToTicks(latency);

    // gem5 deletes this event after it runs (the final `true` argument).
    schedule(new EventFunctionWrapper(
                 [this, instruction_id] {
                     instructionLatencyElapsed(instruction_id);
                 },
                 name() + ".instruction_latency", true),
             deadline);

    DPRINTF(AMX, "Instruction %llu latency deadline is tick %llu\n",
            static_cast<unsigned long long>(instruction_id),
            static_cast<unsigned long long>(deadline));
}

void
AmxAccl::instructionLatencyElapsed(uint64_t instruction_id)
{
    // Mark the wait as finished and continue the opcode's completion path.
    AmxInst *instruction = findInstruction(instruction_id);
    if (!instruction || instruction->state != AmxInst::State::Executing) {
        return;
    }

    instruction->latencyElapsed = true;
    DPRINTF(AMX, "Instruction %llu reached its execution latency\n",
            static_cast<unsigned long long>(instruction_id));

    completeInstructionIfReady(instruction_id);
}

} // namespace gem5
