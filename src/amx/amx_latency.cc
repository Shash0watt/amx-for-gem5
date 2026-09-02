#include <vector>

#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

// Mutiple amx instructions can run at the same time. each takes a certain
// number of cycles to finsih. Instead of creating a seperate event for each
// instruction we use one shared latencyEvent and a list of deadline

// 1. we queue a deadline scheduleInstructionLatency()
// 2. we schedule an event at the earliest deadline armLatencyEvent()
// 3. we handle the wakeups processLatencyEvent()
//     - this is importatnt because we can have multiple events ie loads and a
//     dotproduct finsih at the same time, so we collect all instructions whos
//     deadlines have arrived
// 4. we finish the instruction execution through instructionLatencyElapsed()
// which marks the instruction as complete and checkes if it's ready to reture


void
AmxAccl::scheduleInstructionLatency(uint64_t instruction_id, Cycles latency)
{
    // Record when this instruction has waited for its required AMX latency.
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction,
             "AMX scheduled latency for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));

    // Measure from the issue time, not from a later clock edge.
    const Tick deadline = instruction->issueTick + cyclesToTicks(latency);
    latencyDeadlines.emplace(deadline, instruction_id);

    DPRINTF(AMX, "Instruction %llu latency deadline is tick %llu\n",
            static_cast<unsigned long long>(instruction_id),
            static_cast<unsigned long long>(deadline));
    armLatencyEvent();
}

void
AmxAccl::armLatencyEvent()
{
    // Keep one event scheduled for the earliest instruction deadline.
    if (latencyDeadlines.empty()) {
        if (latencyEvent.scheduled()) {
            deschedule(latencyEvent);
        }
        return;
    }

    const Tick next_deadline = latencyDeadlines.begin()->first;
    panic_if(next_deadline < curTick(),
             "AMX latency deadline was scheduled in the past");

    if (!latencyEvent.scheduled()) {
        schedule(latencyEvent, next_deadline);
    } else if (latencyEvent.when() != next_deadline) {
        reschedule(latencyEvent, next_deadline, true);
    }
}

void
AmxAccl::processLatencyEvent()
{
    // Process every instruction whose required latency has now elapsed.
    // Remove them first because completion may issue new instructions.
    std::vector<uint64_t> elapsed_instructions;
    auto deadline = latencyDeadlines.begin();
    while (deadline != latencyDeadlines.end() &&
           deadline->first <= curTick()) {
        elapsed_instructions.push_back(deadline->second);
        deadline = latencyDeadlines.erase(deadline);
    }

    for (const uint64_t instruction_id : elapsed_instructions) {
        instructionLatencyElapsed(instruction_id);
    }

    armLatencyEvent();
}

void
AmxAccl::instructionLatencyElapsed(uint64_t instruction_id)
{
    // Mark the wait as finished and continue the opcode's completion path.
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction, "AMX latency elapsed for unknown instruction %llu",
             static_cast<unsigned long long>(instruction_id));
    panic_if(instruction->state != AmxInst::State::Executing,
             "AMX latency elapsed for an inactive instruction");
    if (!instruction || instruction->state != AmxInst::State::Executing) {
        return;
    }

    instruction->latencyElapsed = true;
    DPRINTF(AMX, "Instruction %llu reached its execution latency\n",
            static_cast<unsigned long long>(instruction_id));

    completeInstructionIfReady(instruction_id);
}

} // namespace gem5
