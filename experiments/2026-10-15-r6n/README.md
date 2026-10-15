# r6n: r6m with Engram rows v2

r6m (experiments/2026-10-14-r6m) was not promoted: against r6k it had the same bits,
decode 1.2% faster on the geometric mean of the grid and up to 4.5% at small row
counts, but every prefill size 10-13% slower and the 72- and 96-row decode steps
1.5-2.3% slower. The step back of 2026-10-15 found the cause: its Engram gather polled
the control block in host memory from every block, about 100 ms per table of an 8K
chunk ([engram-rows-v2](../2026-10-15-engram-rows-v2/README.md)).

r6n is r6m with:

1. **Engram rows v2** (vLLM 0145, rewritten): one native worker reads the tables in
   layer order; in the forward one thread waits for the table's step (bounded, and
   stopped at once by a failed read) and the TileLang gather loads only this rank's
   rows. Against B12X on the same data the gather is within about 1 us at 16-96
   tokens and faster from 512 tokens, bits equal; the two tables' reads are level at
   decode sizes and faster at prefill sizes.
2. **sparknet 3bc8d10** (`timeout-stop`): r6m's fail-stop collectives, plus the
   fail-stop probe bounding its dispatch by its own registration, and the trace
   clock's TileLang kernel fixed (every traced runtime failed setup on 22c6caa).

Unchanged from r6m: no fallback in the RoCE collective policy (0142), a stopped
collective named as the cause and host rendezvous at startup (0143), and the draft
budget run eagerly (0144), which alone removes the first-request crash's trigger.
Nothing in r6n depends on the TP size; TP3 uses the same settings.

Source: vLLM tree `0c60314d` (patch head `b1309b4a`, as the build replays it), sparknet
`3bc8d10` (tree `f72599fa`); B12X, NCCL, TileLang and TileKernels as r6m.

## Runs

1. [image.json](image.json): build on dgx4, copy, gate (temperature-0 determinism,
   16 and 8 distinct streams).
2. [qualify.json](qualify.json): the Engram, draft-budget and JIT-monitor unit tests
   from the image ([bundles/engram-tests](bundles/engram-tests), dgx4); the collective
   probe; sparknet's fail-stop scenarios on every rank; then boots, most important
   first because a failed boot ends a run: two with every Engram read delayed 200 ms
   ([r6n-delay.json](r6n-delay.json)), which must wait for their rows, never hang;
   six traced boots that each open with a lone short request (the crash's trigger);
   one boot on an empty TileLang cache.
3. [release.json](release.json): the release suite against r6k (r6n, r6k, r6n again).
   The owner decides promotion.

TP3 follows once these pass: see experiments/2026-10-14-one-recipe ("Releasing TP3").
