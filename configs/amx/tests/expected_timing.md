
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

- The 16-byte-by-4-row load into TMM0 must complete before the dependent store
  can issue because the load writes the tile and the store reads it.
- The store writes four 16-byte payloads at a 32-byte stride. Its first row
  begins at cache-line offset 60, so that row is split across two memory
  requests without modifying adjacent bytes.
- The verification configuration is a full queue barrier. It must not commit
  until every store response has drained, the store latency has elapsed, and
  the store has released its TMM0 reader reservation.
- The wider verification reload traces 36 bytes per row beginning four bytes
  before each stored row. Each row should contain four `0x5a` guard bytes,
  sixteen source bytes, and sixteen untouched `0x5a` gap bytes.
- The source payload for row `r` is the byte sequence starting at
  `0x10 * (r + 1)` and increasing by one through the row.
- The release configuration should commit only after the verification load
  completes.
