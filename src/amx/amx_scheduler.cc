#include "amx/amx_accl.hh"

namespace gem5
{

// -------------------------------------------------------------------------
// Ready-instruction selection
// -------------------------------------------------------------------------

AmxAccl::AmxInst *
AmxAccl::findReadyInstruction()
{
    // from oldest first -> youngest instruction
    for (AmxInst &instruction : instructionQueue) {

        // Make sure that tile_config can only issue if it is the oldest
        if (instruction.opcode == AmxOpcode::Config) {
            // It's only the oldest, when it's at the front of the queue
            const bool is_front = &instruction == &instructionQueue.front();
            // The configuration is ready only if it has not issued, it is the
            // oldest queued instruction, and no instruction is currently
            // reading or writing a tile.
            if (instruction.state == AmxInst::State::Pending && is_front &&
                allTilesIdle()) {
                return &instruction;
            }
            // Stop searching even when this configuration is not ready. If we
            // continued, a younger instruction could incorrectly pass the
            // configuration barrier.
            return nullptr;
        }

        // Executing instructions stay in the queue until completion. We can't
        // return a true for them, but independent younger work may still be
        // ready and can be issued
        if (instruction.state != AmxInst::State::Pending) {
            continue;
        }

        // The scoreboard records tiles being used by instructions that are
        // already executing. Wait if this instruction would conflict with one
        // of those active readers or writers.
        if (hasActiveTileHazard(instruction)) {
            continue;
        }

        // Also wait if this instruction depends on an older instruction that
        // has not completed. This preserves dependencies even when that older
        // instruction is itself still pending and therefore not on the active
        // scoreboard yet.
        if (hasOlderTileHazard(instruction)) {
            continue;
            // TODO: when we connect properly with the O3 CPU we should allow
            // TODO: renaming.. well only if saphire rapids supports it
        }

        // This is the oldest pending instruction with no active or queue-order
        // hazard, so it is safe for the issue loop to execute it.
        return &instruction;
    }
    // The queue is empty, or every pending instruction is currently blocked.
    return nullptr;
}

bool
AmxAccl::hasActiveTileHazard(const AmxInst &instruction) const
{
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        const ScoreboardEntry &scoreboard = tileScoreboard[tile];

        // RAW: this instruction wants to read the tile, but an active older
        // instruction has not finished writing it yet.
        const bool raw = AmxInst::hasRAW(instruction.readsTile(tile),
                                         scoreboard.writeActive);

        // WAW: this instruction wants to write the tile while an active older
        // instruction is still writing it.
        const bool waw = AmxInst::hasWAW(instruction.writesTile(tile),
                                         scoreboard.writeActive);

        // WAR: this instruction wants to write the tile while one or more
        // active older instructions are still reading it.
        const bool war = AmxInst::hasWAR(instruction.writesTile(tile),
                                         scoreboard.readerCount > 0);

        if (raw || war || waw) {
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

        if (older.state == AmxInst::State::Completed) {
            continue;
        }

        // Unlike the activeTileHazard check, these also catch dependencies on
        // an older instruction that has not issued yet.
        const bool raw = instruction.hasRAW(older);
        const bool war = instruction.hasWAR(older);
        const bool waw = instruction.hasWAW(older);

        if (raw || war || waw) {
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
