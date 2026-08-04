#include <cstddef>
#include <cstdint>

#include <gem5/m5ops.h>

struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t columnBytes[16];
    uint8_t rows[16];
};

static_assert(sizeof(TileConfig) == 64);

int
main()
{
    constexpr uint64_t tile = 0;
    constexpr size_t sourceRowBytes = 16;
    constexpr size_t destinationStride = 32;
    constexpr size_t rows = 4;
    constexpr uint8_t joe = 0x5a;

    alignas(64) uint8_t source[rows][sourceRowBytes] = {};
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < sourceRowBytes; ++column) {
            source[row][column] = 0x10 * (row + 1) + column;
        }
    }

    // Start near the end of a cache line so the first row crosses into the
    // next line. We have marked bytes which should reveal writes outside the
    // tile rows.
    alignas(64) uint8_t destinationBuffer[256];
    for (auto &byte : destinationBuffer) {
        byte = joe;
    }
    uint8_t *destination = destinationBuffer + 60;

    alignas(64) TileConfig storeConfig = {};
    storeConfig.paletteId = 1;
    storeConfig.columnBytes[tile] = sourceRowBytes;
    storeConfig.rows[tile] = rows;

    alignas(64) TileConfig verifyConfig = {};
    verifyConfig.paletteId = 1;
    verifyConfig.columnBytes[tile] = 36;
    verifyConfig.rows[tile] = rows;

    alignas(64) TileConfig release = {};

    m5_work_begin(0, 0);

    amx_tile_loadconfig(&storeConfig);
    amx_tile_loadd(tile, source, sizeof(source[0]));

    amx_tile_stored(tile, destination, destinationStride);
    amx_tile_loadconfig(&verifyConfig);

    amx_tile_loadd(tile, destination - 4, destinationStride);
    amx_tile_loadconfig(&release);

    m5_work_end(0, 0);
    m5_exit(750);
    m5_quiesce();

    return 0;
}
