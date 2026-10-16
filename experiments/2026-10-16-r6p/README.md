# r6p: r6o with Engram's prefill product exchanged all-to-all

## Hypothesis

Under prefill sequence parallelism every rank projects all T rows of an Engram layer
with its column block of `wkv`, all-gathers the `[T, 25632]` product and keeps only its
own L rows. On TP3 those all-gathers are 50 ms of a 16K prefill: eight `[4048, 8544]`
gathers on dgx3, the slowest reader, which never waits for a peer
([experiments/2026-10-16-tp3-prefill](../2026-10-16-tp3-prefill/README.md)). Exchanging
only each rank's rows moves a third of the bytes, so it should save about 30 ms of a
16K prefill on TP3 (more on TP4, where a rank keeps a quarter). Outputs should be
identical: the same GEMM, with the same blocks laid in the same column order.

## Intended delta

r6p is r6o ([experiments/2026-10-16-r6o](../2026-10-16-r6o/README.md)) with:

1. **0150.** Each rank sends every other rank that rank's rows of its `wkv` column block,
   in an equal-block all-to-all. This is a new registered `all_to_all` collective:
   PyNCCL grouped send/recv. Each rank lays the blocks it receives side by side in rank
   order. Decode runs on full rows and keeps the all-gather.
2. **0149.** The SP tests' stand-in objects gain the fields the model and MoE runner
   began reading on 2026-10-10 and 2026-10-11. Thirteen tests had failed with
   `AttributeError` since then.

3. **0151, a debug timeline** of each step's Engram rows (`SPARK3_ENGRAM_TIMELINE=1`, off and
   free in serving). See below.

Source: vLLM tree `3b2492eb` (patch head `a0519bf5`).

## Tests (dgx4, r6n's image with the changed files mounted)

- The sequence-parallel tests: 155 of 155. These include each rank's exchanged rows
  against its rows of the gathered product, with 3 and 4 ranks simulated and a padded
  last rank.
- The Engram model tests: 25 of 25. Weight transfer: 6 of 6.

## Runs

1. **Image (dgx4).** Built from main and copied to dgx1-3.
2. [gate.json](gate.json): temperature-0 determinism and greedy output must be
   byte-identical to r6o's and r6n's; then 8 and 4 distinct streams.
3. [profile.json](profile.json): torch-profiled r6p, r6o, r6n and r6p again on a 16K
   prefill, a single-stream decode and an eight-stream window. Every arm uses
   production r6n's DSpark cost curves. A profiling arm's serve arguments otherwise
   give it fresh curves: in r6o's profile the eight-stream steps had different row
   counts from r6n's, which hid any Engram effect on decode. In prefill, the
   `[4048, 8544]` all-gathers should become all-to-alls of a third of the bytes.
4. [release.json](release.json): the release-tp3 suite on r6p, r6n and r6p again. This
   is the release comparison for the whole Engram change set. The owner decides
   promotion.

## The first build (2026-10-16)

The first image (tree `4c53bdf5`) failed its gate boot. The TileLang `engram_gate`
kernel requires packed `kv` rows, and the exchange returned a strided view of the
blocks laid side by side. The B12X gate used by the unit tests accepts strided rows,
so the tests did not see it. The exchange now copies the received blocks into packed
rows, the same single copy, and the test checks that the rows are packed. Reading the
all-to-all's blocks in place in the gate, and sending the last rank's real rows only,
would remove the remaining two copies of about 69 MB per call, roughly 10 ms of a TP3
16K prefill. That is a follow-up.

## Decode wait timeline

In r6o's and r6n's profiled eight-stream windows, table 0's wait was 7-8 ms in about
half the steps (median 3 us). The hash runs 3 ms ahead of the wait, so the reader
published about 10 ms after the GPU's IDs, for a few hundred lookups a rank.
[timeline.json](timeline.json) boots r6p with the debug timeline and runs 8 and 1
distinct streams and the long-line greedy text, without the profiler. Each record
splits a step into queueing, waiting for the GPU's IDs, reading and publishing, next
to the GPU's wait for it.
