#include <filesystem>
#include <fstream>

#include "amx/amx_accl.hh"
#include "amx/dp_math_amx.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

namespace
{

uint64_t
tileRowAddress(const amx::Instruction &instruction, uint8_t row)
{
    return instruction.address + row * instruction.stride;
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Failure reporting
// -------------------------------------------------------------------------

void
AmxAccl::reportInstructionFailure(const AmxInst &instruction,
                                  const char *op_name) const
{
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
        case AmxInst::Failure::InvalidConfig:
            panic("AMX: %s %llu has an invalid configuration: %s. "
                  "Guest #GP delivery is not implemented.",
                  op_name, static_cast<unsigned long long>(instruction.id),
                  instruction.failureReason.c_str());
        case AmxInst::Failure::None:
            return;
    }

    panic("AMX instruction has an unknown failure state");
}

// -------------------------------------------------------------------------
// Tile load execution
// -------------------------------------------------------------------------

void
AmxAccl::executeLoadInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured,
             "AMX tile load issued before tile configuration");

    const uint8_t tile = instruction->destination;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];

    DPRINTF(AMX,
            "Executing tile load %llu for TMM%u (%u rows, %u bytes/row)\n",
            static_cast<unsigned long long>(instruction->id), tile, rows,
            row_bytes);

    beginMemoryInstruction(instruction);
    tileScoreboard[tile].writeActive = true;
    tiles[tile] = {};

    for (uint8_t row = 0;
         row < rows && instruction->failure == AmxInst::Failure::None; ++row) {
        dispatchMemoryRead(
            instruction, tileRowAddress(*instruction, row),
            row_bytes, reinterpret_cast<uint8_t *>(tiles[tile].data[row]));
    }

    finishMemoryDispatch(instruction);
}

// -------------------------------------------------------------------------
// Tile configuration execution
// -------------------------------------------------------------------------

void
AmxAccl::executeConfigInstruction(AmxInst *instruction)
{
    panic_if(instructionQueue.empty() ||
                 &instructionQueue.front() != instruction,
             "AMX tile configuration issued away from the queue front");
    panic_if(!allTilesIdle(),
             "AMX tile configuration issued while a tile is active");

    beginMemoryInstruction(instruction);
    instruction->configData.fill(0);

    dispatchMemoryRead(instruction, instruction->address, TILE_CONFIG_BYTES,
                       instruction->configData.data());
    finishMemoryDispatch(instruction);
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
// State-dump for debugging
// -------------------------------------------------------------------------

void
AmxAccl::executeDumpStateInstruction(AmxInst *instruction)
{
    panic_if(!instruction || instruction->opcode != AmxOpcode::DumpState,
             "AMX state dump execution received the wrong instruction");
    panic_if(instructionQueue.empty() ||
                 &instructionQueue.front() != instruction,
             "AMX state dump issued away from the queue front");
    panic_if(!allTilesIdle(), "AMX state dump issued while a tile is active");

    std::filesystem::create_directories(dumpDirectory);
    const std::filesystem::path output_path =
        std::filesystem::path(dumpDirectory) /
        (instruction->dumpName + ".txt");
    std::ofstream stream(output_path);
    panic_if(!stream, "Could not open AMX state dump %s",
             output_path.string());
    writeStateDump(stream, instruction->dumpName);

    instruction->state = AmxInst::State::Completed;
    const uint64_t completed_id = instruction->id;
    DPRINTF(AMX, "Wrote state dump %llu to %s\n",
            static_cast<unsigned long long>(completed_id),
            output_path.string());
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

// -------------------------------------------------------------------------
// Dot-product execution
// -------------------------------------------------------------------------

void
AmxAccl::executeDotProductInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured,
             "AMX dot product issued before tile configuration");

    amx::validateDotProductOp(currentConfig, instruction->destination,
                              instruction->source1, instruction->source2);

    instruction->state = AmxInst::State::Executing;

    for (int tile = 0; tile < NUM_TILES; ++tile) {
        if (instruction->writesTile(tile)) {
            panic_if(tileScoreboard[tile].writeActive,
                     "AMX issued TDPBF16PS with an active tile writer");
            tileScoreboard[tile].writeActive = true;
        }
        if (instruction->readsTile(tile)) {
            ++tileScoreboard[tile].readerCount;
        }
    }

    DPRINTF(AMX, "Executing TDPBF16PS %llu; result is due at tick %llu\n",
            static_cast<unsigned long long>(instruction->id),
            static_cast<unsigned long long>(instruction->issueTick +
                                            cyclesToTicks(dotProductLatency)));
}

// -------------------------------------------------------------------------
// TILEZERO execution
// -------------------------------------------------------------------------

void
AmxAccl::executeZeroInstruction(AmxInst *instruction)
{
    panic_if(!instruction || instruction->opcode != AmxOpcode::Zero,
             "AMX TILEZERO execution received the wrong instruction");
    panic_if(!tilesConfigured,
             "AMX TILEZERO issued before tile configuration");

    const uint8_t tile = instruction->destination;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];
    panic_if(rows == 0 || row_bytes == 0,
             "AMX TILEZERO target tile %u is not configured", tile);

    instruction->state = AmxInst::State::Executing;
    panic_if(tileScoreboard[tile].writeActive,
             "AMX TILEZERO issued with an active tile writer");
    tileScoreboard[tile].writeActive = true;

    DPRINTF(AMX,
            "Executing TILEZERO %llu for TMM%u; result is due at tick %llu\n",
            static_cast<unsigned long long>(instruction->id), tile,
            static_cast<unsigned long long>(instruction->issueTick +
                                            cyclesToTicks(zeroLatency)));
}

// -------------------------------------------------------------------------
// Tile store execution
// -------------------------------------------------------------------------

void
AmxAccl::executeStoreInstruction(AmxInst *instruction)
{
    panic_if(!instruction || instruction->opcode != AmxOpcode::Store,
             "AMX tile store execution received the wrong instruction");
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

    beginMemoryInstruction(instruction);
    panic_if(tileScoreboard[tile].writeActive,
             "AMX tile store issued with an active tile writer");
    ++tileScoreboard[tile].readerCount;

    for (uint8_t row = start_row;
         row < rows && instruction->failure == AmxInst::Failure::None; ++row) {
        dispatchMemoryWrite(
            instruction, tileRowAddress(*instruction, row),
            row_bytes,
            reinterpret_cast<const uint8_t *>(tiles[tile].data[row]));
    }

    finishMemoryDispatch(instruction);
}

// -------------------------------------------------------------------------
// Unified completion pipeline
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

    finalizeInstruction(instruction); // finish all the bookeeping/commit stuff
                                      // for the instruction
}

void
AmxAccl::finalizeInstruction(AmxInst *instruction)
{
    const uint64_t instruction_id = instruction->id;

    switch (instruction->opcode) {
        case AmxOpcode::Load: {
            tileScoreboard[instruction->destination].writeActive = false;
            resourceTracker.complete(AmxResource::TileLoad);
            reportInstructionFailure(*instruction, "Tile load");
            amx::traceBFloat16Tile(currentConfig, tiles,
                                   instruction->destination);
            break;
        }
        case AmxOpcode::Store: {
            const uint8_t tile = instruction->source1;
            panic_if(tileScoreboard[tile].readerCount <= 0,
                     "AMX tile store completed without an active reader");
            --tileScoreboard[tile].readerCount;
            resourceTracker.complete(AmxResource::TileStore);
            if (instruction->failure == AmxInst::Failure::None) {
                currentConfig.startRow = 0;
            }
            reportInstructionFailure(*instruction, "Tile store");
            DPRINTF(AMX, "Completed tile store %llu from TMM%u\n",
                    static_cast<unsigned long long>(instruction_id), tile);
            break;
        }
        case AmxOpcode::Config: {
            panic_if(
                !allTilesIdle(),
                "AMX tile configuration completed while a tile is active");
            amx::TileConfig candidate = {};
            if (instruction->failure == AmxInst::Failure::None) {
                candidate = amx::decodeTileConfig(instruction->configData);
                if (!amx::validateTileConfig(candidate,
                                             instruction->failureReason)) {
                    instruction->failure = AmxInst::Failure::InvalidConfig;
                }
            }
            reportInstructionFailure(*instruction, "Tile configuration");
            commitTileConfig(candidate);
            DPRINTF(AMX, "Committed tile configuration %llu with palette %u\n",
                    static_cast<unsigned long long>(instruction_id),
                    static_cast<unsigned>(candidate.paletteId));
            break;
        }
        case AmxOpcode::DotProduct: {
            amx::doDotProductBF16(currentConfig, tiles,
                                  instruction->destination,
                                  instruction->source1, instruction->source2);
            amx::traceFloat32Tile(currentConfig, tiles,
                                  instruction->destination);

            for (int tile = 0; tile < NUM_TILES; ++tile) {
                if (instruction->writesTile(tile)) {
                    panic_if(
                        !tileScoreboard[tile].writeActive,
                        "AMX TDPBF16PS completed without an active writer");
                    tileScoreboard[tile].writeActive = false;
                }
                if (instruction->readsTile(tile)) {
                    panic_if(
                        tileScoreboard[tile].readerCount <= 0,
                        "AMX TDPBF16PS completed without an active reader");
                    --tileScoreboard[tile].readerCount;
                }
            }
            resourceTracker.complete(AmxResource::DotProduct);
            DPRINTF(AMX, "Completed TDPBF16PS %llu\n",
                    static_cast<unsigned long long>(instruction_id));
            break;
        }
        case AmxOpcode::Zero: {
            const uint8_t tile = instruction->destination;
            panic_if(!tileScoreboard[tile].writeActive,
                     "AMX TILEZERO completed without an active tile writer");
            tiles[tile] = {};
            amx::traceInt8Tile(currentConfig, tiles, tile);
            tileScoreboard[tile].writeActive = false;
            resourceTracker.complete(AmxResource::TileZero);
            DPRINTF(AMX, "Completed TILEZERO %llu for TMM%u\n",
                    static_cast<unsigned long long>(instruction_id), tile);
            break;
        }
        case AmxOpcode::DumpState:
            break;
    }

    instruction->state = AmxInst::State::Completed;
    eraseInstruction(instruction_id);
    tryIssue();
}

} // namespace gem5
