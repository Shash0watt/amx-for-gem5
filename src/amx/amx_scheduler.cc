#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

// -------------------------------------------------------------------------
// Ready-instruction selection
// -------------------------------------------------------------------------

AmxAccl::AmxInst *
AmxAccl::findReadyInstruction()
{
    // Find the oldest safe instruction, while allowing independent younger
    // work to pass an instruction that is temporarily blocked.

    // Recompute the next pipeline wakeup from this scan. Tile completions
    // already wake the scheduler, so only pipeline availability needs a timer.
    nextIssueRetryCycle.reset();

    // Search in program order, from oldest to youngest.
    for (AmxInst &instruction : instructionQueue) {

        // Full barriers issue only from the front after older tile activity
        // has drained. A blocked barrier also stops all younger work.
        if (instruction.isBarrier()) {
            const bool is_front = &instruction == &instructionQueue.front();
            if (instruction.state == AmxInst::State::Pending && is_front &&
                allTilesIdle()) {
                return &instruction;
            }

            // A blocked configuration also blocks everything behind it.
            return nullptr;
        }

        // Issued work stays in the queue until completion, but it cannot issue
        // a second time.
        if (instruction.state != AmxInst::State::Pending) {
            continue;
        }

        // Do not conflict with tile reads or writes already in progress.
        if (hasActiveTileHazard(instruction)) {
            continue;
        }

        // Also preserve dependencies on older work that has not issued yet.
        if (hasOlderTileHazard(instruction)) {
            // TODO: Consider renaming after connecting to the O3 CPU, if the
            // TODO: modeled Sapphire Rapids behavior supports it.
            continue;
        }

        // A busy pipeline blocks only instructions that use that pipeline.
        const std::optional<AmxResource> resource = issueResource(instruction);
        if (resource && !resourceTracker.canIssue(*resource, curCycle())) {
            // Remember when to retry, then look for work on another pipeline.
            noteResourceRetry(resourceTracker.nextIssueCycle(*resource));
            continue;
        }

        // This instruction has passed every readiness check.
        return &instruction;
    }

    // The queue is empty or every remaining instruction is blocked.
    return nullptr;
}

std::optional<AmxAccl::AmxResource>
AmxAccl::issueResource(const AmxInst &instruction) const
{
    // Map each opcode to the independent pipeline that accepts it.
    switch (instruction.opcode) {
        case AmxOpcode::Config:
        case AmxOpcode::DumpState:
            return std::nullopt;
        case AmxOpcode::Load:
            return AmxResource::TileLoad;
        case AmxOpcode::DotProduct:
            return AmxResource::DotProduct;
        case AmxOpcode::Zero:
            return AmxResource::TileZero;
        case AmxOpcode::Store:
            return AmxResource::TileStore;
    }

    panic("AMX instruction has no issue-resource mapping");
}

Cycles
AmxAccl::instructionLatency(const AmxInst &instruction) const
{
    // Return the minimum execution time configured for this opcode.
    switch (instruction.opcode) {
        case AmxOpcode::Load:
            return loadLatency;
        case AmxOpcode::DotProduct:
            return dotProductLatency;
        case AmxOpcode::Zero:
            return zeroLatency;
        case AmxOpcode::Store:
            return storeLatency;
        case AmxOpcode::Config:
            return configLatency;
        case AmxOpcode::DumpState:
            break;
    }

    panic("AMX instruction has no generic latency");
}

void
AmxAccl::noteResourceRetry(Cycles retry_cycle)
{
    // Keep the earliest cycle when any blocked pipeline becomes available.
    if (!nextIssueRetryCycle || retry_cycle < *nextIssueRetryCycle) {
        nextIssueRetryCycle = retry_cycle;
    }
}

void
AmxAccl::updateIssueRetryEvent()
{
    // Make the scheduler's wakeup event match the earliest required retry.
    if (!nextIssueRetryCycle) {
        if (issueRetryEvent.scheduled()) {
            deschedule(issueRetryEvent);
        }
        return;
    }

    const Cycles now = curCycle();
    panic_if(*nextIssueRetryCycle <= now,
             "AMX scheduler requested a retry that is not in the future");
    const Tick retry_tick = clockEdge(*nextIssueRetryCycle - now);

    if (!issueRetryEvent.scheduled()) {
        schedule(issueRetryEvent, retry_tick);
    } else if (issueRetryEvent.when() != retry_tick) {
        reschedule(issueRetryEvent, retry_tick, true);
    }

    DPRINTF(AMX, "Next resource-blocked issue retry is cycle %llu\n",
            static_cast<unsigned long long>(*nextIssueRetryCycle));
}

void
AmxAccl::processIssueRetryEvent()
{
    // A pipeline should now be available, so scan the queue again.
    DPRINTF(AMX, "Retrying resource-blocked AMX instructions\n");
    tryIssue();
}

bool
AmxAccl::hasActiveTileHazard(const AmxInst &instruction) const
{
    // Check the candidate against tile readers and writers already executing.
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
    // Check the candidate against all older unfinished queue entries.
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
    // Configuration is safe only when no tile is being read or written.
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
    // Find the live queue entry referenced by an asynchronous callback.
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
    // Remove one fully completed instruction from the live queue.
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
