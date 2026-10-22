#include <filesystem>
#include <fstream>

#include "amx/amx_accl.hh"
#include "amx/dp_math_amx.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

// Every instruction follows the same lifecycle:
// 1. executeInstruction(): shared issue bookkeeping, then execute<Op>()
// 2. execute<Op>(): check preconditions, then start the work (send memory
//    requests or copy operands)
// 3. completeInstructionIfReady(): waits for memory and the latency
// 4. commitResults(): apply the architectural effect and free the pipeline
//
// DumpState is the exception: it is a debug tool that finishes inside
// executeDumpStateInstruction() without a latency or a commit.

// -------------------------------------------------------------------------
// Issue
// -------------------------------------------------------------------------

void
AmxAccl::executeInstruction(AmxInst *instruction)
{
    // Track issue time
    const Cycles now = curCycle();
    instruction->issueTick = curTick();

    // resource utilization tracking
    const std::optional<AmxResource> resource = issueResource(*instruction);
    if (resource) {
        // mark the resource as used and schedule the nextIssueCycle = now + latency
        // nextIssueCycle is the last check in findReadyInstruction
        // it helps model the throughput values
        resourceTracker.issue(*resource, now);
        const amx::ResourceState &state = resourceTracker.state(*resource);
        DPRINTF(AMX,
                "Issued instruction %llu on %s at cycle %llu; next issue "
                "cycle %llu, %u in flight\n",
                static_cast<unsigned long long>(instruction->id),
                amx::resourceName(*resource),
                static_cast<unsigned long long>(now),
                static_cast<unsigned long long>(state.nextIssueCycle),
                state.inFlight);
    }

    if (instruction->opcode != AmxOpcode::DumpState) {
        // This is to model the completion deadline (aka latency).
        // this uses some helper logic and the gem5 queue to call the functions
        // that 'commit' the results of a instruction
        instruction->latencyElapsed = false;
        scheduleInstructionLatency(instruction->id,
                                   instructionLatency(*instruction));
        instruction->state = AmxInst::State::Executing;
    }

    switch (instruction->opcode) {
        case AmxOpcode::Load:
            executeLoadInstruction(instruction);
            return;
        case AmxOpcode::Store:
            executeStoreInstruction(instruction);
            return;
        case AmxOpcode::Config:
            executeConfigInstruction(instruction);
            return;
        case AmxOpcode::DotProduct:
            executeDotProductInstruction(instruction);
            return;
        case AmxOpcode::Zero:
            executeZeroInstruction(instruction);
            return;
        case AmxOpcode::DumpState:
            executeDumpStateInstruction(instruction);
            return;
    }

    panic("AMX scheduler selected an unknown opcode");
}

// -------------------------------------------------------------------------
// Completion
// -------------------------------------------------------------------------

void
AmxAccl::completeInstructionIfReady(uint64_t instruction_id)
{
    AmxInst *instruction = findInstruction(instruction_id);
    if (!instruction || instruction->state != AmxInst::State::Executing) {
        return;
    }

    const bool is_memory_op = instruction->opcode == AmxOpcode::Load ||
                              instruction->opcode == AmxOpcode::Store ||
                              instruction->opcode == AmxOpcode::Config;

    if (is_memory_op) {
        if (!instruction->translationDispatchComplete ||
            instruction->outstandingTranslations != 0 ||
            instruction->outstandingRequests != 0) {
            return;
        }
        instruction->memoryComplete = true;
    }

    // Failed instructions can finish immediately without waiting for latency
    if (instruction->failure == AmxInst::Failure::None &&
        !instruction->latencyElapsed) {
        return;
    }

    commitResults(instruction);
}

// Memory failures are recorded by the asynchronous translation and response
// callbacks and reported here, once all of the instruction's memory work has
// drained.
void
AmxAccl::reportInstructionFailure(const AmxInst &instruction) const
{
    const char *op_name = "AMX instruction";
    switch (instruction.opcode) {
        case AmxOpcode::Load:
            op_name = "Tile load";
            break;
        case AmxOpcode::Store:
            op_name = "Tile store";
            break;
        case AmxOpcode::Config:
            op_name = "Tile configuration";
            break;
        default:
            break;
    }

    switch (instruction.failure) {
        case AmxInst::Failure::Translation:
            panic("AMX: %s %llu failed address translation: %s. "
                  "Asynchronous fault delivery is not implemented.",
                  op_name, static_cast<unsigned long long>(instruction.id),
                  instruction.fault->name());
        case AmxInst::Failure::MemoryError:
            panic("AMX: %s %llu received an error response.", op_name,
                  static_cast<unsigned long long>(instruction.id));
        case AmxInst::Failure::MissingData:
            panic("AMX: %s %llu received a response without data.", op_name,
                  static_cast<unsigned long long>(instruction.id));
        case AmxInst::Failure::None:
            return;
    }

    panic("AMX instruction has an unknown failure state");
}

void
AmxAccl::commitResults(AmxInst *instruction)
{
    const uint64_t instruction_id = instruction->id;

    reportInstructionFailure(*instruction);

    switch (instruction->opcode) {
        case AmxOpcode::Load: {
            const uint8_t tile = instruction->destination;
            tiles[tile] = instruction->loadData;
            currentConfig.startRow = 0;
            resourceTracker.complete(AmxResource::TileLoad);
            amx::traceBFloat16Tile(currentConfig, tiles, tile);
            DPRINTF(AMX, "Completed tile load %llu into TMM%u\n",
                    static_cast<unsigned long long>(instruction_id), tile);
            break;
        }
        case AmxOpcode::Store: {
            // The rows were already written to memory during execution.
            currentConfig.startRow = 0;
            resourceTracker.complete(AmxResource::TileStore);
            DPRINTF(AMX, "Completed tile store %llu from TMM%u\n",
                    static_cast<unsigned long long>(instruction_id),
                    instruction->source1);
            break;
        }
        case AmxOpcode::Config: {
            // The configuration can only be validated once it has been read.
            const amx::TileConfig config =
                amx::decodeTileConfig(instruction->configData);
            std::string reason;
            panic_if(!amx::validateTileConfig(config, reason),
                     "AMX: Tile configuration %llu is invalid: %s. "
                     "Guest #GP delivery is not implemented.",
                     static_cast<unsigned long long>(instruction_id),
                     reason.c_str());
            commitTileConfig(config);
            DPRINTF(AMX, "Committed tile configuration %llu with palette %u\n",
                    static_cast<unsigned long long>(instruction_id),
                    static_cast<unsigned>(config.paletteId));
            break;
        }
        case AmxOpcode::DotProduct: {
            amx::doDotProductBF16(currentConfig, instruction->destination,
                                  instruction->source2,
                                  tiles[instruction->destination],
                                  instruction->source1Data,
                                  instruction->source2Data);
            resourceTracker.complete(AmxResource::DotProduct);
            amx::traceFloat32Tile(currentConfig, tiles,
                                  instruction->destination);
            DPRINTF(AMX, "Completed TDPBF16PS %llu\n",
                    static_cast<unsigned long long>(instruction_id));
            break;
        }
        case AmxOpcode::Zero: {
            const uint8_t tile = instruction->destination;
            tiles[tile] = {};
            resourceTracker.complete(AmxResource::TileZero);
            amx::traceInt8Tile(currentConfig, tiles, tile);
            DPRINTF(AMX, "Completed TILEZERO %llu for TMM%u\n",
                    static_cast<unsigned long long>(instruction_id), tile);
            break;
        }
        case AmxOpcode::DumpState:
            panic("AMX state dumps finish at execution, not commit");
    }

    instruction->state = AmxInst::State::Completed;
    eraseInstruction(instruction_id);
    tryIssue();
}

// -------------------------------------------------------------------------
// Tile load
// -------------------------------------------------------------------------

void
AmxAccl::executeLoadInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured,
             "AMX tile load issued before tile configuration");

    const uint8_t tile = instruction->destination;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];
    const uint8_t start_row = currentConfig.startRow;

    DPRINTF(AMX,
            "Executing tile load %llu for TMM%u (%u rows, %u bytes/row, "
            "start row %u)\n",
            static_cast<unsigned long long>(instruction->id), tile, rows,
            row_bytes, start_row);

    // Rows land in loadData and only reach the tile register at commit.
    instruction->loadData = {};
    sendReadRequest(instruction, start_row, rows, row_bytes,
                    reinterpret_cast<uint8_t *>(instruction->loadData.data));
}

// -------------------------------------------------------------------------
// Tile store
// -------------------------------------------------------------------------

void
AmxAccl::executeStoreInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured, "AMX store issued before tile configuration");

    const uint8_t tile = instruction->source1;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];
    const uint8_t start_row = currentConfig.startRow;

    DPRINTF(AMX,
            "Executing tile store %llu from TMM%u (%u rows, %u bytes/row, "
            "start row %u)\n",
            static_cast<unsigned long long>(instruction->id), tile, rows,
            row_bytes, start_row);

    sendWriteRequest(instruction, start_row, rows, row_bytes,
                     reinterpret_cast<const uint8_t *>(tiles[tile].data));
}

// -------------------------------------------------------------------------
// Tile configuration
// -------------------------------------------------------------------------

void
AmxAccl::executeConfigInstruction(AmxInst *instruction)
{
    // Nothing to check yet: the configuration is validated at commit, once
    // its 64 bytes have been read.
    instruction->configData.fill(0);
    sendReadRequest(instruction, 0, 1, TILE_CONFIG_BYTES,
                    instruction->configData.data());
}

void
AmxAccl::commitTileConfig(const amx::TileConfig &config)
{
    if (config.paletteId == 0) {
        currentConfig = {};
        tilesConfigured = false;
    } else {
        currentConfig = config;
        tilesConfigured = true;
    }

    // Both palette transitions architecturally clear all tile data.
    amx::clearTiles(tiles);
}

// -------------------------------------------------------------------------
// Dot product
// -------------------------------------------------------------------------

void
AmxAccl::executeDotProductInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured,
             "AMX dot product issued before tile configuration");
    amx::validateDotProductOp(currentConfig, instruction->destination,
                              instruction->source1, instruction->source2);

    // Read the sources now, as the TMUL does at dispatch. Once Executing, the
    // scheduler no longer treats them as read, so younger loads may overwrite
    // the source tiles (no WAR stall for the full dot-product latency).
    instruction->source1Data = tiles[instruction->source1];
    instruction->source2Data = tiles[instruction->source2];

    DPRINTF(AMX, "Executing TDPBF16PS %llu; result is due at tick %llu\n",
            static_cast<unsigned long long>(instruction->id),
            static_cast<unsigned long long>(instruction->issueTick +
                                            cyclesToTicks(dotProductLatency)));
}

// -------------------------------------------------------------------------
// TILEZERO
// -------------------------------------------------------------------------

void
AmxAccl::executeZeroInstruction(AmxInst *instruction)
{
    const uint8_t tile = instruction->destination;
    panic_if(!tilesConfigured,
             "AMX TILEZERO issued before tile configuration");
    panic_if(currentConfig.startRow != 0,
             "AMX: start_row must be 0 for arithmetic operations");
    panic_if(currentConfig.rows[tile] == 0 ||
                 currentConfig.columnBytes[tile] == 0,
             "AMX TILEZERO target tile %u is not configured", tile);

    // Nothing to start: the tile is cleared at commit.
    DPRINTF(AMX,
            "Executing TILEZERO %llu for TMM%u; result is due at tick %llu\n",
            static_cast<unsigned long long>(instruction->id), tile,
            static_cast<unsigned long long>(instruction->issueTick +
                                            cyclesToTicks(zeroLatency)));
}

// -------------------------------------------------------------------------
// State dump for debugging
// -------------------------------------------------------------------------

void
AmxAccl::executeDumpStateInstruction(AmxInst *instruction)
{
    // A barrier, so every older instruction has already completed.
    std::filesystem::create_directories(dumpDirectory);
    const std::filesystem::path output_path =
        std::filesystem::path(dumpDirectory) /
        (instruction->dumpName + ".txt");
    std::ofstream stream(output_path);
    panic_if(!stream, "Could not open AMX state dump %s",
             output_path.string());
    writeStateDump(stream, instruction->dumpName);

    const uint64_t completed_id = instruction->id;
    DPRINTF(AMX, "Wrote state dump %llu to %s\n",
            static_cast<unsigned long long>(completed_id),
            output_path.string());
    instruction->state = AmxInst::State::Completed;
    eraseInstruction(completed_id);
}

void
AmxAccl::writeStateDump(std::ostream &stream,
                        const std::string &dump_name) const
{
    stream << "AMX STATE DUMP: " << dump_name << "\n\n"
           << "Simulation\n"
           << "  Tick        : " << curTick() << '\n'
           << "  Cycle       : " << curCycle() << '\n'
           << "  Accelerator : " << name() << "\n\n"
           << "Tile Configuration\n"
           << "  Configured  : " << (tilesConfigured ? "yes" : "no") << '\n'
           << "  Palette ID  : "
           << static_cast<unsigned>(currentConfig.paletteId) << '\n'
           << "  Start Row   : "
           << static_cast<unsigned>(currentConfig.startRow) << "\n\n"
           << "Tile Registers (values shown as BF16)\n";

    for (int tile = 0; tile < NUM_TILES; ++tile) {
        const uint8_t rows = currentConfig.rows[tile];
        const uint16_t column_bytes = currentConfig.columnBytes[tile];

        if (!tilesConfigured || rows == 0 || column_bytes == 0) {
            stream << "\nTMM" << tile << " - UNCONFIGURED\n"
                   << "  Rows         : " << static_cast<unsigned>(rows)
                   << '\n'
                   << "  Column bytes : " << column_bytes << '\n';
            continue;
        }

        stream << "\nTMM" << tile << " - ACTIVE\n"
               << "  Rows         : " << static_cast<unsigned>(rows) << '\n'
               << "  Column bytes : " << column_bytes << '\n';
        amx::traceBFloat16Tile(stream, currentConfig, tiles, tile);
    }

    stream << '\n';
}

} // namespace gem5
