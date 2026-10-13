# Retiring B12X's WO: WO-A as DeepSeek's reference, in TileLang (port B1 on r6j)

Owner, 2026-10-13: match the reference, quantize only where the reference quantizes,
correctness first and then speed; work until TileLang is well ahead and the non-TileLang
version can be retired.

## Where the reference quantizes

DeepSeek's `inference/model.py` (in the checkpoint) quantizes an activation only inside
`linear()` for FP8 and FP4 weights (`act_quant(x, 32, "ue8m0", float8_e8m0fnu)`, then
`fp8_gemm`). The attention's WO-A is the exception: `wo_a` is declared BF16
(`convert.py` dequantizes it) and applied to the BF16 attention output with an einsum.
WO-B is a `linear()`, so its MXFP8 activations are the reference's. Production (B12X's
fused `wo_projection`) quantizes WO-A's activations too: the last known deviation from
the reference.

[check_cast_reference.py](check_cast_reference.py) compares our activation cast
(TileKernels `per_token_cast`, E4M3, 32-element UE8M0 blocks) with the reference's own
`act_quant` from the checkpoint's `kernel.py`, byte for byte: random data at six
magnitudes, heavy tails, all-zero blocks, blocks below the reference's `1e-4` amax floor,
BF16 subnormals, block maxima on exact powers of two, values at the clamp, and 1-200
rows. Run on dgx4 in the r6j image (2026-10-13): **identical everywhere**, so every FP8
projection already quantizes exactly as the reference, and a cast fused into a GEMM stays
the reference's arithmetic if it keeps these steps.

## The change

vLLM branch `wo-bf16-r6j` (on r6j's `151ea817a`): the `r6d-wo-bf16` work of
[2026-10-11-wo-bf16](../2026-10-11-wo-bf16/README.md) and its sequence-parallel fix,
cherry-picked unchanged, plus its two kernels named in the scheme (`gemm_fp8w_bf16`,
`gemm_fp8w_bf16_partials`):

- WO-A as a grouped W8A16 GEMM: each block-32 FP8 weight tile expanded exactly to BF16
  (E4M3 value times its UE8M0 power of two) and multiplied with the BF16 activations on
  BF16 tensor cores, FP32 accumulation; four K shards, summed in a fixed order whatever
  the path, so a row's bits do not depend on the batch.
- WO-B with the reference's MXFP8 activations; sequence-parallel steps through the
  rank-order projection.
- The attention runs `TileLangWOProjection` under the TileLang family instead of B12X's
  fused kernel.

The output moves toward the reference (new temperature-0 references).

## Windows

- [kernels.json](kernels.json): the WO tests and [bench_wo.py](bench_wo.py) against
  B12X's fused WO (error against FP64 no worse, batch invariance, repeatability, time per
  call warm and cold at every decode and prefill size; fails if TileLang is more than 3%
  slower anywhere). Its own spec, so a speed failure does not stop the serving screen.
- [screen.json](screen.json): r6j + WO ([wo.json](wo.json), the vLLM files mounted over
  the r6j image) / r6j ([r6j.json](r6j.json)) / r6j + WO again: determinism `--wide`
  (each arm identical within itself), quality, the paired sweep, four, eight and sixteen
  different prompts, prefill at 32K and 256K.
- [sweep2.json](sweep2.json): [sweep_wo_a_v2.py](sweep_wo_a_v2.py), WO-A's tiles at four
  shards that the first sweep did not try: 96- and 128-row tiles at 72-128 rows, and
  wider tiles, `block_K` 128, three stages and 256 threads from 256 rows.
