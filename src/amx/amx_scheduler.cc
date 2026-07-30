#include "amx/amx_accl.hh"

namespace gem5
{

// -------------------------------------------------------------------------
// Ready-instruction selection
// -------------------------------------------------------------------------

AmxAccl::AmxInst *
AmxAccl::findReadyInstruction()
{
    for (AmxInst &instruction : instructionQueue) {
        // A configuration is a full barrier: it waits for older work and
        // prevents all younger work from passing it.
        if (instruction.opcode == AmxOpcode::Config) {
            const bool is_front = &instruction == &instructionQueue.front();
            if (instruction.state == AmxInst::State::Pending && is_front &&
                allTilesIdle()) {
                return &instruction;
            }
            return nullptr;
        }

        if (instruction.state != AmxInst::State::Pending) {
            continue;
        }

        if (!hasActiveTileHazard(instruction) &&
            !hasOlderTileHazard(instruction)) {
            return &instruction;
        }
    }

    return nullptr;
}

bool
AmxAccl::hasActiveTileHazard(const AmxInst &instruction) const
{
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        const ScoreboardEntry &scoreboard = tileScoreboard[tile];

        if (instruction.readsTile(tile) && scoreboard.writeActive) {
            return true;
        }
        if (instruction.writesTile(tile) &&
            (scoreboard.writeActive || scoreboard.readerCount > 0)) {
            return true;
        }
    }

    return false;
}

bool
AmxAccl::hasOlderTileHazard(const AmxInst &instruction) const
{
    for (const AmxInst &older : instructionQueue) {
        if (&older == &instruction) {
            return false;
        }
        if (older.state != AmxInst::State::Completed &&
            instruction.hasTileHazardWith(older)) {
            return true;
        }
    }

    panic("AMX scheduler checked an instruction outside its queue");
}

bool
AmxAccl::allTilesIdle() const
{
    for (const ScoreboardEntry &entry : tileScoreboard) {
        if (entry.writeActive || entry.readerCount != 0) {
            return false;
        }
    }

    return true;
}

// -------------------------------------------------------------------------
// Queue lookup and removal
// -------------------------------------------------------------------------

AmxAccl::AmxInst *
AmxAccl::findInstruction(uint64_t instruction_id)
{
    for (AmxInst &instruction : instructionQueue) {
        if (instruction.id == instruction_id) {
            return &instruction;
        }
    }

    return nullptr;
}

void
AmxAccl::eraseInstruction(uint64_t instruction_id)
{
    for (auto it = instructionQueue.begin(); it != instructionQueue.end();
         ++it) {
        if (it->id == instruction_id) {
            instructionQueue.erase(it);
            return;
        }
    }

    panic("AMX tried to erase unknown instruction %llu",
          static_cast<unsigned long long>(instruction_id));
}

} // namespace gem5
