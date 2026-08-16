#ifndef __CONFIGS_AMX_TESTS_TILE_CONFIG_HH__
#define __CONFIGS_AMX_TESTS_TILE_CONFIG_HH__

#include <cstddef>
#include <cstdint>

// The 64-byte memory layout consumed by AMX_TILE_LOADCONFIG.
struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

#endif // __CONFIGS_AMX_TESTS_TILE_CONFIG_HH__
