
### Running tests
```bash
build/X86/gem5.opt configs/amx/tb.py \ --binary configs/amx/binaries/<test_name>
```

### tile_load_timing_test
- The two tile loads should issue 8 cycles apart.
- Each load should complete 45 cycles after it issues, as long as its memory
  responses have already completed.
- Both loads should be in flight at the same time because they write different
  tiles.
- TMM0 should contain 11 and TMM1 should contain 22.
- The release configuration should commit after both loads complete.

### tile_dpbf16_timing_test

- The three tile loads should issue 8 cycles apart.
- The dot product should issue only after all three loads complete.
- The dot product should complete 52 cycles after it issues.
- The final FP32 value in TMM0 should be 24.
- The release configuration should commit after the dot product completes.

### tile_zero_timing_test

- The three tile loads should issue 8 cycles apart.
- `TILEZERO TMM0` should issue after the load to TMM0 completes and complete
  16 cycles after it issues.
- The dependent dot product must not issue until `TILEZERO` completes.
- The final FP32 value in TMM0 should be 23: zeroing removes the initial 1,
  leaving `(2 * 4) + (3 * 5)` rather than 24.
- The release configuration should commit after the zero and dot product
  complete.

### tile_store_timing_test

- TMM0 starts at 1, while TMM1 and TMM2 contain the BF16 pairs `(2, 3)` and
  `(4, 5)`.
- The dot product produces `1 + (2 * 4) + (3 * 5) = 24` in TMM0.
- The dependent store must wait for the dot product to finish, then write 24
  to `storedResult`.
- Reloading the configuration acts as a queue barrier and clears the tiles. It
  must wait until the store has reached memory.
- The final tile load reads `storedResult` into TMM3, which should contain 24.
- The release configuration should commit only after the TMM3 load completes.
