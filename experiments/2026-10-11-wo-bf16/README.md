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
