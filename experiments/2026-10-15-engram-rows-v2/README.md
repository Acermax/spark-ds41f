# 2026-10-15: Engram rows v2

r6m read Engram rows with a native worker (no GIL, no CUDA calls) and gathered them
in the forward. It had r6k's bits and decode at small row counts 3-4.5% faster, but
every prefill size 10-13% slower and the 72- and 96-row decode steps 1.5-2.3% slower
(experiments/2026-10-14-r6m, release 2026-10-15). It is not promoted.

The step back (2026-10-15) compared B12X's Engram path (r6k), antirez's ds4 and r6m,
and timed r6m's parts against B12X's on the same data
([prefill bench](../2026-10-15-engram-io/README.md#prefill-2026-10-15)):
- r6m's gather polled the control block in host memory from every (head, token)
  block: about 12 us a token and table, 100 ms per table of an 8K prefill chunk. That
  is the whole regression. Without the poll the gather matches or beats B12X's lookup
  (709 against 776 us at 8192 tokens), with the same bits.
- The reader is within 6-8% of B12X's at prefill sizes.
- B12X's lookup reads only this rank's rows (about a quarter at TP4) and B12X's
  reader reads the two tables one after the other, so the first Engram layer's rows
  are never held back by the second's. r6m lost both.
- On random rows an 8K chunk reads about 146K 4 KiB blocks, 600 MB and 100 ms per
  table and rank. The release suite's prefill sends real text (the Python standard
  library's source, with a fresh cache salt), so its numbers include Engram I/O on
  real n-grams.

## v2 (vLLM branch `engram-io-v2`, on r6m's 0145)

Kept from r6m: the native reader (no GIL, so no Python thread on the GPU's critical
path), fail-stop on a stalled or failed read, the B12X-free TileLang gather, one path
(no asynchronous or synchronous mode, no resident scales).

Changed:
- **One wait per table and step.** A one-thread TileLang kernel waits for the
  table's step before the gather, bounded by the timeout. It also reads the reader's
  phase, so a failed read stops the GPU at once with its record instead of at the
  timeout. No gather block touches the control block.
- **This rank's rows only.** The gather loads a slot only when its row is in this
  rank's shard and writes zeros otherwise; the reader no longer zero-fills the other
  ranks' slots (about 75% at TP4).
- **Tables in layer order.** One worker serves every reader's steps in the order the
  model stages them, so the second table never competes with the first for the disk.

Not changed: what the reader reads (two planes, O_DIRECT, 4 KiB blocks), the TileLang
hash, the all-reduce and the projection. Nothing depends on the TP size: a rank reads
its share of each step's lookups, so TP3 and TP4 use the same settings.

## Measurement before an image ([v2.json](v2.json), [bundles/v2](bundles/v2))

On the r6m image with v2's files overlaid, dgx4:
- the reader and gather tests, including other ranks' slots filled with E4M3 NaN
  (never read), two tables published in staging order, and a reader failure that
  stops the GPU without waiting for the timeout;
- the gather (wait included) against B12X's lookup at 16-8192 tokens, every row local
  and a quarter local: at most 3 us slower at any size;
- a step's two tables from staging: table 0's rows no later than B12X's serial reader
  by more than 5%.

Only if both pass: an image, the gate, qualify's boots and the release against r6k.
