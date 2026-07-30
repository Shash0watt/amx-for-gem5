#include "amx/amx_accl.hh"

#include <limits>

#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

// -------------------------------------------------------------------------
// Architectural instruction entry points
// -------------------------------------------------------------------------

void
AmxAccl::startAmxLoad(ThreadContext *tc, uint64_t destination, uint64_t source,
                      uint64_t stride)
{
    panic_if(!tc, "AMX: Tile load requires a valid thread context");
    panic_if(destination >= NUM_TILES,
             "AMX: Target tile %llu exceeds max tiles!",
             static_cast<unsigned long long>(destination));

    // give each instruction a unique id
    const uint64_t id = nextInstructionId++;

    // add instruction to queue
    instructionQueue.push_back(AmxInst::tileLoad(
        id, tc, static_cast<uint8_t>(destination), source, stride));

    DPRINTF(AMX, "Queued tile load %llu for TMM%llu\n",
            static_cast<unsigned long long>(id),
            static_cast<unsigned long long>(destination));

    // we just added something, try to issue it
    tryIssue();
}

void
AmxAccl::startAmxLoadConfig(ThreadContext *tc, uint64_t config_address)
{
    panic_if(!tc, "AMX: Tile configuration requires a thread context");
    panic_if(config_address > std::numeric_limits<uint64_t>::max() -
                                  (TILE_CONFIG_BYTES - 1),
             "AMX: Tile configuration address wraps around");
    // uid generation
    const uint64_t id = nextInstructionId++;

    // add to queue
    instructionQueue.push_back(AmxInst::tileConfig(id, tc, config_address));

    DPRINTF(AMX, "Queued tile configuration %llu\n",
            static_cast<unsigned long long>(id));

    // added so try and issue
    tryIssue();
}

// -------------------------------------------------------------------------
// Issue and high-level execution flow
// -------------------------------------------------------------------------

void
AmxAccl::tryIssue()
{
    // An immediate translation callback can re-enter the issue path.
    if (issuingInstructions) {
        // this prevents recursive calling
        // multiple instructions can still issue at the same 'simulation time'
        return;
    }
    
    issuingInstructions = true;

    while (AmxInst *instruction = findReadyInstruction()) {
        // There is no issue-bandwidth model yet, so issue all ready work.
        executeInstruction(instruction);
    }
    issuingInstructions = false;

    if (!instructionQueue.empty()) {
        DPRINTF(AMX, "No additional AMX instruction is ready to issue\n");
    }
}

void
AmxAccl::executeInstruction(AmxInst *instruction)
{
    panic_if(!instruction || instruction->state != AmxInst::State::Pending,
             "AMX scheduler selected an invalid queue entry");
    panic_if(!cpu, "AMX instruction issued without an attached CPU");

    switch (instruction->opcode) {
        case AmxOpcode::Config:
            executeConfigInstruction(instruction);
            return;
        case AmxOpcode::Load:
            executeLoadInstruction(instruction);
            return;
        case AmxOpcode::Compute:
            executeComputeInstruction(instruction);
            return;
        case AmxOpcode::Store:
            executeStoreInstruction(instruction);
            return;
    }

    panic("AMX scheduler selected an unknown opcode");
}

} // namespace gem5
