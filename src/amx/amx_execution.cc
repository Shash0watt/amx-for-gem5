#include <limits>

#include "amx/amx_accl.hh"
#include "amx/dp_math_amx.hh"
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

    // mark the start of the instruction
    beginMemoryInstruction(instruction);
    tileScoreboard[tile].writeActive = true;
    tiles[tile] = {};

    // make the split memory requests
    for (uint8_t row = 0;
         row < rows && instruction->failure == AmxInst::Failure::None; ++row) {
        dispatchMemoryRead(instruction,
                           tileRowAddress(*instruction, row, row_bytes),
                           row_bytes, MemoryTarget::TileRow, tile, row);
    }

    // mark the end of making requests
    finishMemoryDispatch(instruction);
}

void
AmxAccl::completeLoadIfReady(uint64_t instruction_id)
{
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction || instruction->opcode != AmxOpcode::Load,
             "AMX load completion checked the wrong instruction");

    // A load is ready only after both its memory work and latency are done.
    if (!instruction->memoryComplete || !instruction->latencyElapsed) {
        return;
    }

    completeLoadInstruction(instruction);
}

void
AmxAccl::completeLoadInstruction(AmxInst *instruction)
{
    // Release everything owned by a load that is fully ready to complete.
    panic_if(!instruction || instruction->opcode != AmxOpcode::Load,
             "AMX load completion received the wrong instruction");
    panic_if(!instruction->translationDispatchComplete ||
                 instruction->outstandingTranslations != 0 ||
                 instruction->outstandingRequests != 0,
             "AMX tile load completed with outstanding memory work");
    panic_if(!instruction->memoryComplete || !instruction->latencyElapsed,
             "AMX tile load completed before memory and latency elapsed");

    instruction->state = AmxInst::State::Completed;
    tileScoreboard[instruction->destination].writeActive = false;
    resourceTracker.complete(AmxResource::TileLoad);
    reportLoadFailure(*instruction);

    amx::traceBFloat16Tile(currentConfig, tiles, instruction->destination);

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

    pendingConfigInstructionId = 0;
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
// Dot-product execution
// -------------------------------------------------------------------------

void
AmxAccl::executeDotProductInstruction(AmxInst *instruction)
{
    // Start a dot product and keep its tiles busy until its latency expires.
    panic_if(!tilesConfigured,
             "AMX dot product issued before tile configuration");

    // Confirm that the three configured tile shapes can be multiplied.
    amx::validateDotProductOp(
        currentConfig, instruction->destination, instruction->source1,
        instruction->source2);

    instruction->state = AmxInst::State::Executing;

    // Record every tile this operation reads or writes. This prevents other
    // instructions from changing those tiles before the result is ready.
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
            static_cast<unsigned long long>(
                instruction->issueTick + cyclesToTicks(dotProductLatency)));
}

void
AmxAccl::finishDotProductInstruction(uint64_t instruction_id)
{
    // Produce the delayed result, then release the tiles and pipeline state.
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction || instruction->opcode != AmxOpcode::DotProduct,
             "AMX TDPBF16PS completion received the wrong instruction");
    panic_if(!instruction->latencyElapsed,
             "AMX TDPBF16PS completed before its latency elapsed");

    // The tile data changes here, after the modeled latency has elapsed.
    amx::doDotProductBF16(
        currentConfig, tiles, instruction->destination, instruction->source1,
        instruction->source2);
    amx::traceFloat32Tile(currentConfig, tiles, instruction->destination);

    // Remove the read and write reservations made when the operation started.
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        if (instruction->writesTile(tile)) {
            panic_if(!tileScoreboard[tile].writeActive,
                     "AMX TDPBF16PS completed without an active writer");
            tileScoreboard[tile].writeActive = false;
        }
        if (instruction->readsTile(tile)) {
            panic_if(tileScoreboard[tile].readerCount <= 0,
                     "AMX TDPBF16PS completed without an active reader");
            --tileScoreboard[tile].readerCount;
        }
    }

    instruction->state = AmxInst::State::Completed;

    // Record that this operation is no longer using the dot-product pipeline.
    resourceTracker.complete(AmxResource::DotProduct);
    DPRINTF(AMX, "Completed TDPBF16PS %llu\n",
            static_cast<unsigned long long>(instruction_id));

    // Remove the completed work and look for instructions it was blocking.
    eraseInstruction(instruction_id);
    tryIssue();
}

// -------------------------------------------------------------------------
// TILEZERO execution
// -------------------------------------------------------------------------

void
AmxAccl::executeZeroInstruction(AmxInst *instruction)
{
    // Check that TILEZERO has a valid destination in the current layout.
    panic_if(!instruction || instruction->opcode != AmxOpcode::Zero,
             "AMX TILEZERO execution received the wrong instruction");
    panic_if(!tilesConfigured, "AMX TILEZERO issued before tile configuration");
    panic_if(instruction->destination < 0 ||
                 instruction->destination >= NUM_TILES,
             "AMX TILEZERO has invalid destination tile %d",
             instruction->destination);

    const uint8_t tile = instruction->destination;
    const uint16_t rows = currentConfig.rows[tile];
    const uint16_t row_bytes = currentConfig.columnBytes[tile];
    panic_if(rows == 0 || row_bytes == 0,
             "AMX TILEZERO target tile %u is not configured", tile);
    panic_if(rows > MAX_ROWS || row_bytes > MAX_COLS_BYTES,
             "AMX TILEZERO target tile %u has invalid configured dimensions",
             tile);

    // Keep other instructions from using the tile until zeroing finishes.
    instruction->state = AmxInst::State::Executing;
    panic_if(tileScoreboard[tile].writeActive,
             "AMX TILEZERO issued with an active tile writer");
    tileScoreboard[tile].writeActive = true;

    DPRINTF(AMX,
            "Executing TILEZERO %llu for TMM%u; result is due at tick %llu\n",
            static_cast<unsigned long long>(instruction->id), tile,
            static_cast<unsigned long long>(
                instruction->issueTick + cyclesToTicks(zeroLatency)));
}

void
AmxAccl::finishZeroInstruction(uint64_t instruction_id)
{
    // Finish the delayed operation once its required latency has passed.
    AmxInst *instruction = findInstruction(instruction_id);
    panic_if(!instruction || instruction->opcode != AmxOpcode::Zero,
             "AMX TILEZERO completion received the wrong instruction");
    panic_if(instruction->state != AmxInst::State::Executing,
             "AMX TILEZERO completion received an inactive instruction");
    panic_if(!instruction->latencyElapsed,
             "AMX TILEZERO completed before its latency elapsed");

    const uint8_t tile = instruction->destination;
    panic_if(tile >= NUM_TILES,
             "AMX TILEZERO completion has invalid destination tile %u", tile);
    panic_if(!tileScoreboard[tile].writeActive,
             "AMX TILEZERO completed without an active tile writer");

    // Clear the entire tile, including rows and columns outside its layout.
    tiles[tile] = {};
    amx::traceInt8Tile(currentConfig, tiles, tile);

    // Release the tile and TILEZERO pipeline now that the result is ready.
    tileScoreboard[tile].writeActive = false;
    instruction->state = AmxInst::State::Completed;
    resourceTracker.complete(AmxResource::TileZero);

    DPRINTF(AMX, "Completed TILEZERO %llu for TMM%u\n",
            static_cast<unsigned long long>(instruction_id), tile);

    // Remove the completed work and retry instructions it was blocking.
    eraseInstruction(instruction_id);
    tryIssue();
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
