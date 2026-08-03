#ifndef __AMX_AMX_TILE_HH__
#define __AMX_AMX_TILE_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace gem5
{

namespace amx
{

inline constexpr int MaxRows = 16;
inline constexpr int MaxColumnBytes = 64;
inline constexpr int NumTiles = 8;
inline constexpr size_t TileConfigBytes = 64;

using RawTileConfig = std::array<uint8_t, TileConfigBytes>;

struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[NumTiles];
    uint8_t reserved1[16];
    uint8_t rows[NumTiles];
    uint8_t reserved2[8];
};

static_assert(sizeof(TileConfig) == TileConfigBytes);
static_assert(offsetof(TileConfig, columnBytes) == 16);
static_assert(offsetof(TileConfig, reserved1) == 32);
static_assert(offsetof(TileConfig, rows) == 48);
static_assert(offsetof(TileConfig, reserved2) == 56);

struct TileRegister
{
    int8_t data[MaxRows][MaxColumnBytes];
};

using TileRegisterFile = std::array<TileRegister, NumTiles>;

TileConfig decodeTileConfig(const RawTileConfig &raw);
bool validateTileConfig(const TileConfig &config, std::string &reason);
void clearTiles(TileRegisterFile &tiles);

void traceInt8Tile(const TileConfig &config, const TileRegisterFile &tiles,
                   uint8_t tile_idx);
void traceInt32Tile(const TileConfig &config, const TileRegisterFile &tiles,
                    uint8_t tile_idx);
void traceFloat32Tile(const TileConfig &config, const TileRegisterFile &tiles,
                      uint8_t tile_idx);

} // namespace amx
} // namespace gem5

#endif // __AMX_AMX_TILE_HH__
