
# TILELOAD

## Current Impementation 

- [x] it performs a strided load into the tile strcut within the accelerator
- [] uses 2 dcache ports from the cpu core to load the tile (in gem5 we actually just have one port and configure it to allow parallel reads to model the resource contention)
- [] it is restartable upon page faults
- [] tiles outside the configuration are zeroed out

*Right now only the 2nd version of the intrinsic is implemented (because it is what I needed to write an optimised gemm)

## Ports in Saphire Rapids
- The L1 data cache can service two 512-bit loads every cycle, and the L2 has a 64 byte per cycle link to L1
- The L1 cache is now wider with 3 load ports instead of 2, and deeper with larger Load and Store Buffers.

- 3 load capable ports (2,3, 11)
- 2 store ports (4, 9) 
- and also ports for addressing

- AMX tile-memory operations are synchronous, coherent with CPU memory operations, and run in multi-cycle execution units,

Modeling AMX as consuming at most two 64-byte tile-row loads per cycle is a reasonable Sapphire Rapids model because it matches the 128-byte-per-cycle L1D load bandwidth.


## Tile Load Intrinsics
there are two tile load intrinsics
### void __tile_loadd (__tile1024i* dst, const void* base, size_t stride)
"Load tile rows from memory specifieid by base address and stride into destination tile dst. The shape of tile is specified in the struct of __tile1024i. The register of the tile is allocated by compiler."

here the compiler handles it ie:\
```C
__tile1024i a = {16, 64}; // give us any tile availible, configured to be 16x64
__tile_loadd(&a, ptr, stride);
```
### void _tile_loadd (constexpr int dst, const void * base, size_t stride)
"Load tile rows from memory specifieid by base address and stride into destination tile dst using the tile configuration previously configured via _tile_loadconfig."

the difference between these two functions is who manages the tile regsiter asignments and tile configuration

heree ourselves have to configure the tile:\
```C
_tile_loadconfig(&cfg); // cfg defines tmm2's shape
_tile_loadd(2, ptr, stride);
```



