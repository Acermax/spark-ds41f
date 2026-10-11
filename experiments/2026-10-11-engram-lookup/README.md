# Port B10: the Engram row lookup in TileLang (r6d)

The TileLang migration's B10 (see the
[inventory](../2026-10-09-tilelang-migration/README.md)). Each Engram layer's lookup
turns the step's hashed table rows into BF16 embeddings:
- 24 heads of 256 E4M3 values a token;
- one E8M0 scale byte per 32 values.

r6d runs B12X's Triton kernel for it: one program per token and head, on the side
stream that waits for the disk reader. Production reads the compact rows the io_uring
reader staged in mapped host memory.

## The change (vLLM branch `r6d-engram-lookup`, on r6d's tree)

- `b54e2edae` (module): `tilelang/engram_lookup.py`.
  - Contract: the same as B12X's, with compact or shard-indexed rows, resident or
    compact scales, zeros for dead slots and rows outside the shard, and the tail
    cleared on request.
  - Arithmetic: the same. E4M3 widens exactly; the E8M0 byte becomes a power of two,
    with byte 0 as 2^-127 and 255 as NaN; the product rounds once to BF16.
  - Layout: sixteen values a thread, one 16-byte load and two 16-byte stores.
- `b9afe0275` (switch): under the TileLang family, every lookup decodes with the
  TileLang kernel: the asynchronous disk path, the synchronous one and resident
  tables. The disk reader stays B12X's (migration step 7). The kernel compiles when
  the lookup is prepared, never during graph capture.

## Window 1 ([w1.json](w1.json))

1. Kernel bundle: the tests compare the kernel bit for bit with B12X's, in all three
   layouts, with dead slots and shard edges, and under graph replays with the live
   count changed. Then [bench_engram_lookup.py](bench_engram_lookup.py):
   - bits and time per call against B12X's kernel at decode (1-96 tokens) and prefill
     (512-8192) sizes, warm and cold;
   - tables in mapped host memory (production's disk path) and on the device;
   - replay repeatability.

   It fails on any bit or any size more than 3% slower.
2. Lean screen bracketed by r6d with `token_determinism.py --wide`. The bits equal
   B12X's, so the outputs must equal r6d's.

Acceptance: identical outputs and references to r6d; the bench faster than B12X
everywhere (the owner's bar); serving level or better.

### Window 1 kernel results (2026-10-11, 11:23 UTC)

Tests 13 passed: bits equal to B12X's in every layout. The bench missed the bar at small
sizes:

| tokens | disk (mapped host) warm vs B12X | resident warm vs B12X |
|---|---|---|
| 1-48 | 1.08-1.31x slower (0.3-0.5 us) | 1.03-1.61x slower (0.6-0.7 us) |
| 72-96 | 0.94-1.02x | 0.91-0.99x |
| 512-8192 | 0.89-1.00x faster | 0.79-0.99x faster |

The first version issued three dependent global loads: the live count, then the hash,
then the row.

vLLM `91eb87dcf` issues every load at once, with the same bits:
- a compact slot's row is its own position;
- a shard row is clamped to 0 when the slot is not local;
- dead slots select zero afterwards.

Window 2 ([w2.json](w2.json)) is the kernel bundle again, tests and bench, kernel only.
The serving screen of window 1 stands: the bits are unchanged.
