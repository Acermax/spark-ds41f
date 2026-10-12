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
