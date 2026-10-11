# WO with BF16 WO-A activations, as the reference (port B1, on r6d)

Owner, 2026-10-11: compute at native precision, never lower; speed never below B12X.
DeepSeek's reference dequantizes `wo_a` to BF16 (`convert.py`) and projects the BF16
attention output with an einsum; production's B12X `wo_projection` and the earlier
TileLang port both quantized it to MXFP8 first. WO-B's MXFP8 activations are the
reference's own (`linear`).

## The change (vLLM branch `r6d-wo-bf16`, on r6d's tree)

- `78edfe9e0` (module): `fp8w_bf16_gemm`, a grouped W8A16 GEMM (every group in one
  launch): each block-32 FP8 weight tile is expanded exactly to BF16 (an E4M3 value
  times its UE8M0 power of two) and multiplied on BF16 tensor cores with FP32
  accumulation. `TileLangWOProjection` runs WO-A with it on the BF16 activations; row
  tiles of 16-128 share one K loop, so a row's bits do not depend on the batch. This
  also drops the separate activation cast that made the earlier port slow at 512-2048
  rows.
- `0bf1d6260` (switch, port B1): the attention's WO runs `TileLangWOProjection` under
  the TileLang family instead of B12X's fused kernel.

## Window 1 ([w1.json](w1.json))

The WO tests in the r6d image (WO-A within one BF16 rounding of FP64, far inside what
MXFP8 activations give; batch-invariant) and B1's bench against B12X's fused WO (error
no worse, invariance, time per call warm and cold at every decode and prefill size;
fails if TileLang is more than 3% slower anywhere), then the lean screen bracketed by
r6d, with the sixteen-prompt acceptance sweep
([precision-acceptance](../2026-10-11-precision-acceptance/README.md)) because the
change moves temperature-0 text.

Acceptance: temperature-0 identical within the arm; the bench at or above B12X at every
size; decode and prefill at or above r6d.

### Window 1 (2026-10-11, 09:04 UTC): did not compile

The kernel bundle failed (`test_wo_projection`, `test_wo_a_keeps_bf16_activations`:
"1 error detected in the compilation of tvm_kernels.cu"), and the window was closed
early by request before the WO arm booted; production kept serving r6d. A CPU-only
compile of the four row tiles in the r6d image (no GPU, memory-capped) reproduced it:
the 16- and 32-row tiles (`block_N=32`) compile, the 64- and 128-row tiles
(`block_N=64`) do not. The kernel read one UE8M0 exponent per weight element; at that
width the gather vectorizes to an eight-byte pack that TVM writes as an `int`
expression assigned to `uint2`.

vLLM `d9bf17e78` decodes each tile's exponents (at most 2x2) into FP32 powers of two in
shared memory once per K tile and multiplies from there: the same products, one more
barrier per tile. All four tiles compile; the generated loop double-buffers the scale
tile with the pipeline. Window 2 ([w2.json](w2.json)) is window 1 again on the fixed
arm, with the acceptance sweep.

### Window 2 (2026-10-11, 11:13 UTC): not repeatable and slower

Tests passed (9). The bench failed on two counts:
- **Repeatability:** graph replays at 1-48 rows gave other bits. The producer warps'
  scale stores raced the consumers.
- **Speed:** the whole WO took 46 against B12X's 30.8 us warm at decode, and was slower
  up to 2048 rows (0.75x at 8192). The error against FP64 was below B12X's
  (0.49/0.081 against 0.68/0.114, max/RMS).

The window was closed before the serving arms. The fixes, measured in the
[sweep](../2026-10-11-wo-bf16-sweep/README.md):
- vLLM `145a64d10` turns warp specialization off. Every config is repeatable.
- vLLM `50f6f886d` takes the swept tiles: 32 columns and two stages up to 128 rows, 64
  columns past them. WO-A alone takes 19.5-20 us at 1-16 rows, 28.7 at 48, 43 at 96,
  140 at 512, 399 at 2048 and 1504 at 8192.

Window 3 ([w3.json](w3.json), kernel only) reruns the whole-projection bench against
B12X. At decode WO-A alone is about two thirds of B12X's whole WO, so split-K, with a
fixed shard count at every row count, is the likely next step.

### Window 3 (2026-10-11, 13:19 UTC, kernel only)

- **Correctness:** repeatable at every size (0 of 200 replays differ), with error below
  B12X's.
- **Speed against B12X's fused WO:** 1.21-1.41x at 1-8 rows (about 34 against 25 us
  warm), 1.00-1.26x at 12-64, 1.24-1.43x at 72-512, 0.96x at 2048, 0.70x at 8192.
- **Why decode lags:** WO-A's 64-tile grid leaves SMs idle at decode (about 400 GB/s).

vLLM `cec0da371` sums K in four shards at every row count:
- decode rows (up to 128) give each shard its own CTA: FP32 partials, then
  `splitk_reduce`;
- longer steps add the shards inside the kernel in the same order.

A row's bits do not depend on the path; the WO-A test compares row counts across both.
Window 4 ([w4.json](w4.json), kernel only) reruns the bench.

### Window 4 (2026-10-11, 14:00 UTC, kernel only)

- **Correctness:** repeatable at every size (0 of 200 replays), no row differs from its
  full batch, error below B12X's (0.49/0.080 against 0.68/0.114).
- **Speed against B12X's fused WO, warm / cold:**

  | rows | TileLang (us) | B12X (us) | ratio |
  |---|---|---|---|
  | 1-16 | 22.5-24.0 / 101-103 | 22.6-34.4 / 94-97 | 0.69-1.04 / 1.06-1.08 |
  | 24-64 | 28.6-53.8 / 105-111 | 24.5-39.2 / 98-103 | 1.15-1.37 / 1.07-1.08 |
  | 72-96 | 81-95 / 121-125 | 54-61 / 104-106 | 1.51-1.55 / 1.16-1.17 |
  | 512 | 299 / 272 | 214 / 189 | 1.40 / 1.44 |
  | 2048 | 853 / 825 | 808 / 765 | 1.06 / 1.08 |
  | 8192 | 3348 / 3199 | 4524 / 4376 | 0.74 / 0.73 |

- **Reading:** split-K brought warm decode (1-16 rows) level with B12X. Cold is still
  6-8% slower everywhere up to 2048 rows. From 24 rows the split path is slower than
  the unsplit kernel of the earlier sweep (WO-A alone took 43 us at 96 rows unsplit; the
  whole WO now takes 95). The likely cost is the FP32 partials (four shards of every
  output, written then read back: 3 MB each way at 96 rows).

The [split sweep](../2026-10-11-wo-bf16-sweep/README.md#window-3) chooses, per row
bucket, the path (split or in-kernel shards) and the tile, at two and four shards.

### The split sweep's tiles (vLLM `726be4365`)

The [split sweep](../2026-10-11-wo-bf16-sweep/README.md#window-3-results-2026-10-11-1424-utc) chose a path
and a tile per bucket, all at four shards and 64 columns:

| rows | path | tile rows | WO-A alone, warm (us) |
|---|---|---|---|
| 1-16 | split | 16 | 10.4-11.4 (was 13.0-14.1) |
| 17-32 | split | 32 | 13.6-13.9 (was 16.8-17.4) |
| 33-64 | split | 64 | 19.0-19.7 (was 22.4-24.2) |
| 65-96 | in-kernel | 32 | 34.8 at 96 |
| 97-128 | split | 64 | 39.0 at 128 |
| 129+ | in-kernel | 64 | 69 at 256, 131 at 512, 460 at 2048, 1760 at 8192 |

From window 4's whole-projection times, WO-B and the inverse RoPE take about 10 us at
decode but about 160 us at 512 rows and 390 at 2048. There WO-A's gain alone cannot
reach B12X's whole WO (214 and 808 us).

Window 5 ([w5.json](w5.json), kernel only) reruns the bench at `726be4365`. The bench
now times TileLang's parts alone (inverse RoPE, WO-A, WO-B with its activation cast)
and judges the floor per bucket.

### Window 5 (2026-10-11, 14:56 UTC, kernel only)

- **Correctness:** WO tests passed (9). No size differs from its full batch, every
  size is repeatable (0 of 200), and the error is below B12X's.
- **Parts alone, warm (us):**

  | rows | inverse RoPE | WO-A | WO-B with its cast |
  |---|---|---|---|
  | 1-16 | 1.2 | 10.7-11.4 | 8.1-8.6 |
  | 24-64 | 1.2-1.3 | 13.6-19.8 | 10.4-17.8 |
  | 72-96 | 1.4-1.5 | 34.5-34.7 | 32.7-33.1 |
  | 512 | 3.0 | 131 | 94 |
  | 2048 | 7.1 | 458 | 360 |
  | 8192 | 93 | 1916 | 1352 |

- **Whole WO against B12X, warm / cold:** 1.00-1.16 / 1.03-1.05 at 1-16 rows,
  1.07-1.30 / 1.04-1.05 at 24-64, 1.17-1.28 / 1.22-1.25 at 72-96, 1.25 / 1.37 at 512,
  1.05 / 1.07 at 2048, 0.76 / 0.74 at 8192.
- **Not cleaner than window 4.** B12X's own warm time moved by about 20% between the
  windows (29 against 24 us at one row), so the ratios move with the run.
  - TileLang's warm whole call (32 us at one row) is well above the sum of its parts
    (20 us). Back to back, WO-A's and WO-B's weights (8 and 10 MB) likely do not
    both stay in L2.
  - The sweep chose its tiles warm, with WO-A's weight alone in L2. Fewer, wider CTAs
    are worse when part of the weight streams from DRAM.
  - The cold whole call improved (98-100 against 101-103 us at decode).

The kernel bench cannot settle this. Serving can: the L2 prefetch stages each layer's
WO weights ahead of the step. [r6e pieces window 1](../2026-10-11-r6e-candidate/w5.json)
screens r6e + WO-BF16 (vLLM `r6e-wobf16` `25d1f3cf2`: these commits on r6e) against r6e,
with the acceptance sweep because BF16 WO-A changes the bits.
