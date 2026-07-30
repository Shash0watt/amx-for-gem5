#include <limits>

#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

namespace
{

uint64_t
tileRowAddress(const amx::Instruction &instruction, uint8_t row,
               uint16_t row_bytes)
{
    if (row != 0) {
        panic_if(instruction.stride > (std::numeric_limits<uint64_t>::max() -
                                       instruction.address) /
                                          row,
                 "AMX tile load row address wraps around");
    }

    const uint64_t address = instruction.address + row * instruction.stride;
    panic_if(row_bytes != 0 && address > std::numeric_limits<uint64_t>::max() -
                                             (row_bytes - 1),
             "AMX tile load row crosses the address limit");
    return address;
}

void
reportLoadFailure(const amx::Instruction &instruction)
{
    switch (instruction.failure) {
        case amx::Instruction::Failure::Translation:
            panic("AMX: Tile load %llu failed address translation: %s. "
                  "Asynchronous fault delivery is not implemented.",
                  static_cast<unsigned long long>(instruction.id),
                  instruction.fault->name());
        case amx::Instruction::Failure::MemoryError:
            panic("AMX: Tile load %llu received an error response.",
                  static_cast<unsigned long long>(instruction.id));
        case amx::Instruction::Failure::MissingData:
            panic("AMX: Tile load %llu received a response without data.",
                  static_cast<unsigned long long>(instruction.id));
        case amx::Instruction::Failure::InvalidConfig:
            panic("AMX: Tile load %llu has an internal configuration failure.",
                  static_cast<unsigned long long>(instruction.id));
        case amx::Instruction::Failure::None:
            return;
    }

    panic("AMX tile load has an unknown failure state");
}

void
reportConfigFailure(const amx::Instruction &instruction)
{
    switch (instruction.failure) {
        case amx::Instruction::Failure::Translation:
            panic("AMX: Tile configuration %llu failed address translation: "
                  "%s. Asynchronous fault delivery is not implemented.",
                  static_cast<unsigned long long>(instruction.id),
                  instruction.fault->name());
        case amx::Instruction::Failure::MemoryError:
            panic("AMX: Tile configuration %llu received an error response.",
                  static_cast<unsigned long long>(instruction.id));
        case amx::Instruction::Failure::MissingData:
            panic("AMX: Tile configuration %llu received a response without "
                  "data.",
                  static_cast<unsigned long long>(instruction.id));
        case amx::Instruction::Failure::InvalidConfig:
            panic("AMX: Invalid tile configuration %llu: %s. Guest #GP "
                  "delivery is not implemented.",
                  static_cast<unsigned long long>(instruction.id),
                  instruction.failureReason.c_str());
        case amx::Instruction::Failure::None:
            return;
    }

    panic("AMX tile configuration has an unknown failure state");
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Tile load execution
// -------------------------------------------------------------------------

void
AmxAccl::executeLoadInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured,
             "AMX tile load issued before tile configuration");
    panic_if(instruction->destination < 0 ||
                 instruction->destination >= NUM_TILES,
             "AMX tile load has invalid tile %d", instruction->destination);
    panic_if(!instruction->threadContext,
             "AMX tile load has no thread context");

    const uint8_t tile = instruction->destination;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];
    panic_if(rows > MAX_ROWS || row_bytes > MAX_COLS_BYTES,
             "AMX tile %d has invalid configured dimensions", tile);

    DPRINTF(AMX,
            "Executing tile load %llu for TMM%u (%u rows, %u bytes/row)\n",
            static_cast<unsigned long long>(instruction->id), tile, rows,
            row_bytes);

    beginMemoryInstruction(instruction);
    tileScoreboard[tile].writeActive = true;
    tiles[tile] = {};

    for (uint8_t row = 0;
         row < rows && instruction->failure == AmxInst::Failure::None; ++row) {
        dispatchMemoryRead(instruction,
                           tileRowAddress(*instruction, row, row_bytes),
                           row_bytes, MemoryTarget::TileRow, tile, row);
    }

    finishMemoryDispatch(instruction);
}

void
AmxAccl::finishLoadInstruction(AmxInst *instruction)
{
    panic_if(!instruction || instruction->opcode != AmxOpcode::Load,
             "AMX load completion received the wrong instruction");
    panic_if(!instruction->translationDispatchComplete ||
                 instruction->outstandingTranslations != 0 ||
                 instruction->outstandingRequests != 0,
             "AMX tile load completed with outstanding memory work");

    instruction->state = AmxInst::State::Completed;
    tileScoreboard[instruction->destination].writeActive = false;
    reportLoadFailure(*instruction);

    amx::traceInt32Tile(currentConfig, tiles, instruction->destination);

    const uint64_t completed_id = instruction->id;
    eraseInstruction(completed_id);
    tryIssue();
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
    panic_if(!instruction->threadContext,
             "AMX tile configuration has no thread context");

    beginMemoryInstruction(instruction);
    instruction->issueTick = curTick();
    instruction->configData.fill(0);

    dispatchMemoryRead(instruction, instruction->address, TILE_CONFIG_BYTES,
                       MemoryTarget::TileConfig);
    finishMemoryDispatch(instruction);
}

void
AmxAccl::finishConfigInstruction(uint64_t instruction_id)
{
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction || instruction->opcode != AmxOpcode::Config,
             "AMX config completion received the wrong instruction");
    panic_if(!instruction->translationDispatchComplete ||
                 instruction->outstandingTranslations != 0 ||
                 instruction->outstandingRequests != 0,
             "AMX tile configuration completed with outstanding memory work");
    panic_if(!allTilesIdle(),
             "AMX tile configuration completed while a tile is active");

    amx::TileConfig candidate = {};
    if (instruction->failure == AmxInst::Failure::None) {
        candidate = amx::decodeTileConfig(instruction->configData);
        if (!amx::validateTileConfig(candidate, instruction->failureReason)) {
            instruction->failure = AmxInst::Failure::InvalidConfig;
        }
    }

    instruction->state = AmxInst::State::Completed;
    reportConfigFailure(*instruction);
    commitTileConfig(candidate);

    DPRINTF(AMX, "Committed tile configuration %llu with palette %u\n",
            static_cast<unsigned long long>(instruction_id),
            static_cast<unsigned>(candidate.paletteId));

    configCompletionInstructionId = 0;
    eraseInstruction(instruction_id);
    tryIssue();
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
// Compute and store placeholders
// -------------------------------------------------------------------------

void
AmxAccl::executeComputeInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured, "AMX compute issued before tile configuration");
    panic_if(
        instruction->destination < 0 ||
            instruction->destination >= NUM_TILES ||
            instruction->source1 < -1 || instruction->source1 >= NUM_TILES ||
            instruction->source2 < -1 || instruction->source2 >= NUM_TILES,
        "AMX compute has an invalid tile operand");

    instruction->state = AmxInst::State::Executing;
    tileScoreboard[instruction->destination].writeActive = true;

    if (instruction->source1 != -1) {
        tileScoreboard[instruction->source1].readerCount++;
    }
    tileScoreboard[instruction->destination].readerCount++;
    if (instruction->source2 != -1) {
        tileScoreboard[instruction->source2].readerCount++;
    }

    // TODO: implement compute execution and completion.
}

void
AmxAccl::executeStoreInstruction(AmxInst *instruction)
{
    panic_if(!tilesConfigured, "AMX store issued before tile configuration");
    panic_if(instruction->source1 < -1 || instruction->source1 >= NUM_TILES,
             "AMX store has an invalid tile operand");

    instruction->state = AmxInst::State::Executing;
    if (instruction->source1 != -1) {
        tileScoreboard[instruction->source1].readerCount++;
    }

    // TODO: implement store execution and completion.
}

} // namespace gem5
