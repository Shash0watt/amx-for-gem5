#include "amx/tile_amx.hh"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/AMX.hh"

namespace gem5
{

namespace amx
{

namespace
{

constexpr const char *TileBorder = "+=================================="
                                   "======================================+";

bool
allZero(const uint8_t *bytes, size_t size)
{
    return std::all_of(bytes, bytes + size,
                       [](uint8_t byte) { return byte == 0; });
}

void
appendTileHeader(std::ostream &stream, uint8_t tile, uint16_t rows,
                 uint16_t columns, const char *column_unit)
{
    stream << '\n' << TileBorder << '\n';
    stream << "  AMX REGISTER STATE: [ TMM" << static_cast<int>(tile)
           << " ]\n";
    stream << "  Layout Dimensions : " << rows << " Active Rows x " << columns
           << ' ' << column_unit << '\n';
    stream << TileBorder << '\n';
}

void
appendRowLabel(std::ostream &stream, uint8_t row)
{
    stream << " Row [" << std::setw(2) << std::setfill('0') << std::dec
           << static_cast<int>(row) << "]: ";
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Architectural tile configuration
// -------------------------------------------------------------------------

TileConfig
decodeTileConfig(const RawTileConfig &raw)
{
    TileConfig config = {};
    config.paletteId = raw[0];
    config.startRow = raw[1];

    std::copy_n(raw.begin() + 2, sizeof(config.reserved0), config.reserved0);

    for (size_t tile = 0; tile < NumTiles; ++tile) {
        const size_t offset = 16 + tile * 2;
        config.columnBytes[tile] =
            static_cast<uint16_t>(raw[offset]) |
            (static_cast<uint16_t>(raw[offset + 1]) << 8);
    }

    std::copy_n(raw.begin() + 32, sizeof(config.reserved1), config.reserved1);
    std::copy_n(raw.begin() + 48, NumTiles, config.rows);
    std::copy_n(raw.begin() + 56, sizeof(config.reserved2), config.reserved2);
    return config;
}

bool
validateTileConfig(const TileConfig &config, std::string &reason)
{
    const auto reject = [&reason](const std::string &message) {
        reason = message;
        return false;
    };

    if (config.paletteId != 0 && config.paletteId != 1) {
        return reject("unsupported palette " +
                      std::to_string(config.paletteId));
    }
    if (config.startRow != 0) {
        return reject("nonzero start_row is not supported");
    }
    if (!allZero(config.reserved0, sizeof(config.reserved0))) {
        return reject("reserved bytes 2-15 must be zero");
    }
    if (!allZero(config.reserved1, sizeof(config.reserved1))) {
        return reject("reserved bytes 32-47 must be zero");
    }
    if (!allZero(config.reserved2, sizeof(config.reserved2))) {
        return reject("reserved bytes 56-63 must be zero");
    }

    size_t total_bytes = 0;
    for (int tile = 0; tile < NumTiles; ++tile) {
        const uint16_t rows = config.rows[tile];
        const uint16_t columns = config.columnBytes[tile];

        if (rows > MaxRows) {
            return reject("tile " + std::to_string(tile) +
                          " has more than 16 rows");
        }
        if (columns > MaxColumnBytes) {
            return reject("tile " + std::to_string(tile) +
                          " has more than 64 column bytes");
        }
        if ((rows == 0) != (columns == 0)) {
            return reject("tile " + std::to_string(tile) +
                          " has mismatched row and column dimensions");
        }
        if (config.paletteId == 0 && (rows != 0 || columns != 0)) {
            return reject("palette zero requires an empty tile layout");
        }

        total_bytes += rows * columns;
    }

    if (total_bytes > NumTiles * MaxRows * MaxColumnBytes) {
        return reject("tile layout exceeds the 8 kib register file");
    }
    return true;
}

void
clearTiles(TileRegisterFile &tiles)
{
    for (TileRegister &tile : tiles) {
        tile = {};
    }
}

// -------------------------------------------------------------------------
// Debug trace formatting
// -------------------------------------------------------------------------

void
traceInt8Tile(const TileConfig &config, const TileRegisterFile &tiles,
              uint8_t tile_idx)
{
    panic_if(tile_idx >= NumTiles, "AMX printer: tile index %d out of bounds!",
             tile_idx);

    const uint16_t active_rows = config.rows[tile_idx];
    const uint16_t active_columns = config.columnBytes[tile_idx];

    std::ostringstream stream;
    appendTileHeader(stream, tile_idx, active_rows, active_columns,
                     "Column Bytes");

    for (uint8_t row = 0; row < active_rows; ++row) {
        appendRowLabel(stream, row);
        for (uint16_t column = 0; column < active_columns; ++column) {
            const int8_t value = tiles[tile_idx].data[row][column];
            stream << std::setw(4) << std::setfill(' ') << std::dec
                   << static_cast<int>(value) << ' ';
            if ((column + 1) % 4 == 0 && column + 1 < active_columns) {
                stream << "| ";
            }
        }
        stream << '\n';
    }
    stream << TileBorder;

    DPRINTF(AMX, "%s\n", stream.str().c_str());
}

void
traceInt32Tile(const TileConfig &config, const TileRegisterFile &tiles,
               uint8_t tile_idx)
{
    panic_if(tile_idx >= NumTiles, "AMX printer: tile index %d out of bounds!",
             tile_idx);

    const uint16_t active_rows = config.rows[tile_idx];
    const uint16_t active_columns =
        config.columnBytes[tile_idx] / sizeof(int32_t);

    std::ostringstream stream;
    appendTileHeader(stream, tile_idx, active_rows, active_columns,
                     "Column Int32s");

    for (uint8_t row = 0; row < active_rows; ++row) {
        appendRowLabel(stream, row);
        for (uint16_t column = 0; column < active_columns; ++column) {
            int32_t value = 0;
            std::memcpy(&value,
                        &tiles[tile_idx].data[row][column * sizeof(int32_t)],
                        sizeof(value));
            stream << std::setw(8) << std::setfill(' ') << std::dec << value
                   << ' ';
            if ((column + 1) % 4 == 0 && column + 1 < active_columns) {
                stream << "| ";
            }
        }
        stream << '\n';
    }
    stream << TileBorder;

    DPRINTF(AMX, "%s\n", stream.str().c_str());
}

void
traceBFloat16Tile(const TileConfig &config, const TileRegisterFile &tiles,
                  uint8_t tile_idx)
{
    panic_if(tile_idx >= NumTiles, "AMX printer: tile index %d out of bounds!",
             tile_idx);

    const uint16_t active_rows = config.rows[tile_idx];
    const uint16_t active_columns =
        config.columnBytes[tile_idx] / sizeof(uint16_t);

    std::ostringstream stream;
    appendTileHeader(stream, tile_idx, active_rows, active_columns,
                     "Column BF16s");

    for (uint8_t row = 0; row < active_rows; ++row) {
        appendRowLabel(stream, row);
        for (uint16_t column = 0; column < active_columns; ++column) {
            uint16_t bfloat_bits = 0;
            std::memcpy(
                &bfloat_bits,
                &tiles[tile_idx].data[row][column * sizeof(uint16_t)],
                sizeof(bfloat_bits));

            const uint32_t float_bits =
                static_cast<uint32_t>(bfloat_bits) << 16;
            float value = 0.0F;
            std::memcpy(&value, &float_bits, sizeof(value));

            stream << std::setw(10) << std::setfill(' ') << value << ' ';
            if ((column + 1) % 4 == 0 && column + 1 < active_columns) {
                stream << "| ";
            }
        }
        stream << '\n';
    }
    stream << TileBorder;

    DPRINTF(AMX, "%s\n", stream.str().c_str());
}

void
traceFloat32Tile(const TileConfig &config, const TileRegisterFile &tiles,
                 uint8_t tile_idx)
{
    panic_if(tile_idx >= NumTiles, "AMX printer: tile index %d out of bounds!",
             tile_idx);

    const uint16_t active_rows = config.rows[tile_idx];
    const uint16_t active_columns =
        config.columnBytes[tile_idx] / sizeof(float);

    std::ostringstream stream;
    appendTileHeader(stream, tile_idx, active_rows, active_columns,
                     "Column FP32s");

    for (uint8_t row = 0; row < active_rows; ++row) {
        appendRowLabel(stream, row);
        for (uint16_t column = 0; column < active_columns; ++column) {
            float value = 0.0F;
            std::memcpy(&value,
                        &tiles[tile_idx].data[row][column * sizeof(float)],
                        sizeof(value));
            stream << std::setw(10) << std::setfill(' ') << value << ' ';
            if ((column + 1) % 4 == 0 && column + 1 < active_columns) {
                stream << "| ";
            }
        }
        stream << '\n';
    }
    stream << TileBorder;

    DPRINTF(AMX, "%s\n", stream.str().c_str());
}

} // namespace amx
} // namespace gem5
