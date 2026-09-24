# How to do a optimised gemm (mat C = mat A * mat B)

## Data Alignment
- align to 64 byte boundaries
- mat A - normal packing \
- mat B - vinni packing

## Tile mangagment
- minimize LDTILECFG because of it's high latency

- AMX has 8 physical tile regsiters. try and use all the tiles. (ie. 2 C/accumalator tiles, 2 A tiles, 2 B tiles) so that you can take more advantage of parallelism

- In a naive GEMM loop, an A-tile is re-read from memory multiple times across accumulator loops, instead:
1. pre load all A tiles,
2. re use all A tiles accross N accumulators


## SW Pipelining & Interleaving
- Interleave TDP with with LOADS, don't have long sequential blocks of TILELOAD or TILESTORE
- unroll innermost loops to avoid branch and conditional checks

## Cache Hierarchy & Memory Blcoking 

- apparently you can do loads directly from L2 cache.. (but I haven't implemented that.. should I?) -> It would make 

## Avoid LOAD-to-STORE forwarding stalls. AMX Tileload cannot forward data directly to a STORE.
## AMX TDP turns of AVX-512 Port 5 to save power, (Rapdily switching between instructions can cause stalls as port 5 opens and closes)






