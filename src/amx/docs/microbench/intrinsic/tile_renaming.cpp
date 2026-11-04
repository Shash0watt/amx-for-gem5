// Does real hardware rename tile registers? Each test runs the same work
// twice: once reusing one tile, so every iteration has a false (WAW or WAR)
// dependency on the one before, and once rotating through six tiles, so it
// has none. If the two are the same speed, the false dependency costs
// nothing (the tile is renamed, or its old value is read early). If the
// one-tile version is slower, the difference is the stall.
//
//   g++ -O2 -mamx-tile -mamx-bf16 -o tile_renaming tile_renaming.cpp
//   AMX_BENCH_CPU=<core> ./tile_renaming
//
// Pick a core whose SMT sibling is idle, and check that
// /proc/sys/kernel/perf_event_paranoid is 2 or less. See perf_cycles.h for
// how the cycles are counted. Tiles 6 and 7 are the dot-product sources;
// tiles 0-5 are the ones being reused or rotated.
//
// What the gem5 model gives (AmxAccl.py defaults, no renaming):
//   A) zero + dp:   one tile ~68 (52 dp + 16 zero), six tiles ~16
//   B) load + dp:   one tile ~97+ (52 dp + 45 load), six tiles ~16
//   C) store + load: one tile waits for the whole store, memory included

#include <sys/mman.h>

#include <cstdint>
#include <cstring>

#include "perf_cycles.h"

alignas(64) TileConfig config;     // 16 rows x 64 bytes
alignas(64) uint16_t source[512];  // BF16 1.0
alignas(64) uint8_t destination[6 * 1024];

// ---------------------------------------------------------------------------
// A) TILEZERO breaks a dot-product chain. Each zero is a WAW on the dot
// product before it, then the next dot product needs the zero (RAW).
// Without renaming, the zero waits for the dot product to finish, so one
// tile costs dp latency + zero latency per pair. With renaming, every pair
// is independent and only the dot-product issue rate limits it.
// (TILEZERO could also be a special case at rename, like XOR zeroing, which
// is why B repeats this with a load.)
// ---------------------------------------------------------------------------

void
zeroDotProductOneTile(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_zero(0);
        _tile_dpbf16ps(0, 6, 7);
    }
    _mm_lfence();
}

void
zeroDotProductSixTiles(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_zero(0);
        _tile_dpbf16ps(0, 6, 7);
        _tile_zero(1);
        _tile_dpbf16ps(1, 6, 7);
        _tile_zero(2);
        _tile_dpbf16ps(2, 6, 7);
        _tile_zero(3);
        _tile_dpbf16ps(3, 6, 7);
        _tile_zero(4);
        _tile_dpbf16ps(4, 6, 7);
        _tile_zero(5);
        _tile_dpbf16ps(5, 6, 7);
    }
    _mm_lfence();
}

// ---------------------------------------------------------------------------
// B) TILELOADD breaks a dot-product chain: the same as A with a load, the
// "load C, accumulate" pattern. This tells general renaming apart from a
// TILEZERO special case.
// ---------------------------------------------------------------------------

void
loadDotProductOneTile(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_loadd(0, source, 64);
        _tile_dpbf16ps(0, 6, 7);
    }
    _mm_lfence();
}

void
loadDotProductSixTiles(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_loadd(0, source, 64);
        _tile_dpbf16ps(0, 6, 7);
        _tile_loadd(1, source, 64);
        _tile_dpbf16ps(1, 6, 7);
        _tile_loadd(2, source, 64);
        _tile_dpbf16ps(2, 6, 7);
        _tile_loadd(3, source, 64);
        _tile_dpbf16ps(3, 6, 7);
        _tile_loadd(4, source, 64);
        _tile_dpbf16ps(4, 6, 7);
        _tile_loadd(5, source, 64);
        _tile_dpbf16ps(5, 6, 7);
    }
    _mm_lfence();
}

// ---------------------------------------------------------------------------
// C) A load overwrites the tile a store just read (WAR), the "store C, start
// the next C" pattern. The gem5 model holds the load until the store has
// finished, memory included. The cold version stores to lines that are not
// in any cache, so if the one-tile loop waits for the store's memory it
// takes about a DRAM round trip per pair, while the six-tile loop is
// limited by bandwidth. If the two are close, the load only waits for the
// store to read the tile.
// (The one-tile loop also has a real RAW, load -> next store, so in L1 it
// is slower by about the load latency even with no WAR stall.)
// ---------------------------------------------------------------------------

const size_t ColdBytes = size_t(1) << 30;
uint8_t *coldBuffer;
size_t coldOffset = 0;

// The next 1 KiB of a 1 GiB buffer, so the stores never reuse a line.
uint8_t *
nextColdBlock()
{
    uint8_t *block = coldBuffer + coldOffset;
    coldOffset = (coldOffset + 1024) & (ColdBytes - 1);
    return block;
}

void
storeLoadOneTileL1(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_stored(0, destination, 64);
        _tile_loadd(0, source, 64);
    }
    _mm_mfence(); // LFENCE doesn't wait for stores to reach the cache
}

void
storeLoadSixTilesL1(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_stored(0, destination + 0 * 1024, 64);
        _tile_loadd(0, source, 64);
        _tile_stored(1, destination + 1 * 1024, 64);
        _tile_loadd(1, source, 64);
        _tile_stored(2, destination + 2 * 1024, 64);
        _tile_loadd(2, source, 64);
        _tile_stored(3, destination + 3 * 1024, 64);
        _tile_loadd(3, source, 64);
        _tile_stored(4, destination + 4 * 1024, 64);
        _tile_loadd(4, source, 64);
        _tile_stored(5, destination + 5 * 1024, 64);
        _tile_loadd(5, source, 64);
    }
    _mm_mfence();
}

void
storeLoadOneTileCold(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_stored(0, nextColdBlock(), 64);
        _tile_loadd(0, source, 64);
    }
    _mm_mfence();
}

void
storeLoadSixTilesCold(int n)
{
    for (int i = 0; i < n; ++i) {
        _tile_stored(0, nextColdBlock(), 64);
        _tile_loadd(0, source, 64);
        _tile_stored(1, nextColdBlock(), 64);
        _tile_loadd(1, source, 64);
        _tile_stored(2, nextColdBlock(), 64);
        _tile_loadd(2, source, 64);
        _tile_stored(3, nextColdBlock(), 64);
        _tile_loadd(3, source, 64);
        _tile_stored(4, nextColdBlock(), 64);
        _tile_loadd(4, source, 64);
        _tile_stored(5, nextColdBlock(), 64);
        _tile_loadd(5, source, 64);
    }
    _mm_mfence();
}

int
main()
{
    setUpBenchmark();

    fillConfig(config, 16, 64);
    for (int i = 0; i < 512; ++i) {
        source[i] = 0x3f80; // BF16 1.0, so dot products stay normal
    }
    asm volatile("" ::: "memory"); // the intrinsics don't say they read memory
    _tile_loadconfig(&config);
    _tile_loadd(6, source, 64);
    _tile_loadd(7, source, 64);

    // Cycles are per pair (one breaker + one dot product, or one store +
    // one load), so the one-tile and six-tile rows compare directly.
    measure("A) zero + dp, one tile", zeroDotProductOneTile, 1024, 1);
    measure("A) zero + dp, six tiles", zeroDotProductSixTiles, 256, 6);
    measure("B) load + dp, one tile", loadDotProductOneTile, 1024, 1);
    measure("B) load + dp, six tiles", loadDotProductSixTiles, 256, 6);

    measure("C) store + load, one tile, L1", storeLoadOneTileL1, 1024, 1);
    measure("C) store + load, six tiles, L1", storeLoadSixTilesL1, 256, 6);

    // Fault every page in now, not in a timed run. The 1 GiB is far bigger
    // than the LLC, and the stores walk it from the start, which was
    // touched longest ago.
    coldBuffer = static_cast<uint8_t *>(std::aligned_alloc(1 << 21, ColdBytes));
    madvise(coldBuffer, ColdBytes, MADV_HUGEPAGE);
    for (size_t i = 0; i < ColdBytes; i += 4096) {
        coldBuffer[i] = 0;
    }
    measure("C) store + load, one tile, cold", storeLoadOneTileCold, 1024, 1);
    measure("C) store + load, six tiles, cold", storeLoadSixTilesCold, 256, 6);

    std::free(coldBuffer);
    _tile_release();
    return 0;
}
