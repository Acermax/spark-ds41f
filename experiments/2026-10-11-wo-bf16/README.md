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
r6d.

Acceptance: temperature-0 identical within the arm; the bench at or above B12X at every
size; decode and prefill at or above r6d.
