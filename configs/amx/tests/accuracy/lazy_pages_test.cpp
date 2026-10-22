#include <cstdint>
#include <cstdlib>

#include <gem5/m5ops.h>

// AMX must work on heap pages the CPU has never touched. gem5's SE mode maps
// them lazily (the first access page-faults and gets a page), so this used to
// panic with "failed address translation: Page-Fault".

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

int
main()
{
    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 3; ++tile) {
        config.columnBytes[tile] = 64;
        config.rows[tile] = 16;
    }

    alignas(64) static uint8_t pattern[16][64];
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            pattern[row][col] = static_cast<uint8_t>(row * 64 + col + 1);
        }
    }

    // 1 MiB is served by mmap, which SE mode maps lazily. With a 4 KiB
    // stride every row starts 32 bytes before the end of its own untouched
    // page, so it also crosses into the next untouched page.
    const size_t page = 4096;
    uint8_t *heap = static_cast<uint8_t *>(std::malloc(1 << 20));
    uint8_t *aligned = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(heap) + page - 1) & ~(page - 1));
    uint8_t *stored = aligned + page - 32;
    uint8_t *untouched = aligned + 32 * page;

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);

    // TMM0 = pattern, stored into untouched pages, then read back into TMM2.
    amx_tile_loadd(0, pattern, 64);
    amx_tile_stored(0, stored, page);
    amx_tile_loadd(2, stored, page);

    // TMM1 is loaded from yet more untouched pages, so it must be all zeros.
    amx_tile_loadd(1, untouched, page);

    amx_dump_state("lazy_pages");
    amx_wait();
    m5_work_end(0, 0);

    return 0;
}
