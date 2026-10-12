# r6h: the last 0.9% at sixteen streams, at r6g's bits

[r6g's release comparison](../2026-10-12-r6g-release/README.md) left one metric below r6d:
sixteen different prompts at once, -0.9% ±0.7%. At 96 verify rows r6g's step is about
1.6 ms longer than r6d's. Per call (fixed-row profiles, adaptive verification off):

| kernel | r6d | r6g | per step |
| --- | ---: | ---: | ---: |
| mHC `project_streams`, 96 rows (grid 40 x 6, twice a layer) | 30.6 µs | 41.5 µs | +0.9 ms |
| mHC `project_streams`, 48 rows (40 x 3) | 17.8 µs | 22.4 µs | +0.2 ms |
| MoE combine (5 x 96) | 12.0 µs | 16.7 µs | +0.18 ms |
| DSpark Markov projection (505 x 1) | 46.7 µs | 67.3 µs | +0.11 ms |

The largest single kernel is not a regression: the vocabulary head GEMM (grid 1010 x 2,
verification and drafting) takes about 2.17 ms a call on both, 4.5% of a step. Its
packed 12-bit shard (32,320 x 5,120) is about 248 MB, about 115 GB/s, under half the
GB10's DRAM bandwidth; it lives in the display carve-out.

## Window 1 ([w1.json](w1.json)): two configuration levers on the r6g image

No code change, same bits:
- **head**: `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=0`, the embedding and packed head in regular
  memory (about 0.55 GB a rank; TP4's lowest MemAvailable is about 20 GiB);
- **curves**: DSpark cost curves profiled for this image (empty
  `r6g-image-20261012`, real-row profiling) instead of r6's of 2026-10-05; adaptive
  verification sizes each step from them, and the committed tokens do not depend on
  how many drafts a step verifies.

base / head / curves / base again, six samples of sixteen and eight different prompts,
then a fixed-row profile of base and head (the head GEMM a call).

## Window 2 ([w2.json](w2.json)): three kernel changes, bit-identical by construction

vLLM branch `r6h-mhc-fused-split` (on r6g's `fbaa04151`):
- `544ee7703` mHC: fn split into its TF32 parts while loading it
  ([mhc-split](../2026-10-12-mhc-split/README.md): bit-identical, 2-5 µs a call);
- `54af64774` MoE: the folded combine loads the shared row before the routes;
- `7fd70268a` vocabulary heads: token tiles vary fastest. Past 64 rows a head call has
  two token tiles; with vocabulary blocks fastest the second tile re-read the whole
  248 MB shard from DRAM (about 2.2 ms a 96-row call). At sixteen streams the drafter's
  head call always has 80 rows, so every step paid it; r6d has the same kernel.

[bundles/kernels](bundles/kernels) runs the mHC, MoE and vocabulary head tests with the
three files (dgx1); [bundles/head](bundles/head) times the head per call with both grids
and compares the outputs bit for bit (dgx2).

### Window 1 results (2026-10-12, 05:20-05:45 UTC)

| arm | sixteen prompts, tok/s | eight prompts |
| --- | ---: | ---: |
| base (r6g) | 220.1 ±1.5% | 178.4 ±0.4% |
| head out of the carve-out | 220.3 ±1.3% | 178.5 ±0.4% |
| curves profiled for the image | **223.5** ±0.9% | **180.3** ±0.4% |
| base again | 220.2 ±1.4% | 178.4 ±0.4% |

The carve-out streams at full speed: no change. Curves profiled for the image add 1.5%
and 1.1% (adaptive verification had sized r6g's steps from r6's cost shape).

### Window 2 and 3 results (05:45-05:48 UTC)

The mHC, MoE and vocabulary head tests pass with the three changes (82 passed). The
head bench, bit-identical: one tile (up to 64 rows) unchanged at 1.10-1.24 ms a call;
two tiles 2.29-2.38 ms with vocabulary blocks fastest, 1.60-1.89 ms with token tiles
fastest (80 rows: 2.34 to 1.65 ms; the drafter's call at sixteen streams).

## r6h

r6g's series and vLLM `r6h-mhc-fused-split` (0115-0117): patch head `e39cd9da9938`,
tree `68c9abd9c1d7` (replayed as build prepare does). [r6h-tp4.json](r6h-tp4.json) is
r6g-tp4 with the `-r6h` image and its own cost curves (`ring4-r6h-20261012`, profiled
with real rows on the image's first boot). [image.json](image.json) builds and copies it
and gates it (determinism with and without logprobs); [release-tp4.json](release-tp4.json)
runs the release comparison, r6d / r6h / r6d again.

## TP3 ([release-tp3.json](release-tp3.json)), trimmed for a same-morning re-cable

r6d (`r6d-tp3.json`) against r6h ([r6h-tp3.json](r6h-tp3.json): the promoted TP3 profile
with the `-r6h` image and its own cost curves, `tp3-r6h-20261012`, profiled on its first
boot) on the triangle: quality, prefill at 32K / 256K / 500K (two repeats),
determinism `--wide`, the paired acceptance sweep, and four and eight different prompts
at once. The bench's one-prompt decode points and the latency extras are left out.

Arm order (owner, 2026-10-12): the candidate first, so heat soak does not count against
it, and again last for drift. The TP3 comparison had already started in the old order
(r6d first, r6h second), so its r6h numbers are conservative. The TP4 comparison runs
r6h / r6d / r6h again.
