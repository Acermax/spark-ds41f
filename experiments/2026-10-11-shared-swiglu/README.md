# The shared expert's SwiGLU at the reference's precision (r6d)

Owner, 2026-10-11: compute at native precision, never lower; speed never below B12X.
The r6d audit found vLLM's `SiluAndMulWithClamp` (the shared expert's activation)
rounding `silu(gate)` to BF16 before multiplying by `up`; DeepSeek's reference
(`Expert.forward`) computes `silu(gate) * up` in FP32 after the clamps and rounds once.

## The change (vLLM branch `r6d-shared-swiglu`, on r6d's tree)

- `bba9f2ab2` (module): `tilelang/activation.py`, a one-launch TileLang kernel with the
  reference's arithmetic (FP32 clamps, `g / (1 + exp(-g)) * u`, one BF16 rounding), and
  `kernels.shared_expert_activation` behind `SHARED_SWIGLU_FP32` (off).
- `94f08a874` (switch): on under the TileLang family; the decoder layer swaps its shared
  expert's `act_fn`.

## Window 1 ([w1.json](w1.json))

The activation tests in the r6d image (bit for bit against the reference's torch
arithmetic at 1-2024 rows, batch-invariant), then the lean screen bracketed by r6d.

Acceptance: temperature-0 identical within the arm; speed level (one launch, as before).

### Window 1 results (2026-10-11, TP4, r6d / shared SwiGLU / r6d again)

Kernel bundle passed (bit for bit against the reference arithmetic). Temperature 0: the
arm is identical to itself in every scenario (0 token and 0 logprob differences); its
reference digests change (`c5ccdeb1410d` / `861d2583e6a3`), as expected when the product
is no longer rounded twice.

| metric | r6d | r6d again | shared SwiGLU |
|---|---|---|---|
| prose-c1 step | 31.61 ms | 31.60 ms | 31.63 ms |
| json-nothink-c1 step | 36.78 ms | 36.80 ms | 37.21 ms |
| prose-c1 / json-c1 tok/s | 68.39 / 102.09 | 68.42 / 102.17 | 69.72 / 107.16 |
| accepted per draft, prose / json | 1.257 / 3.000 | 1.257 / 3.000 | 1.306 / 3.262 |
| prefill 1K / 16K tok/s | 2649 / 5636 | 2589 / 5562 | 2668 / 5658 |
| c1 distinct / c8 distinct tok/s | 87.30 / 175.66 | 87.04 / 175.43 | 85.50 / 180.54 |

Prose step time is level. JSON's step is 1.2% longer on a different text (acceptance
3.262 against 3.000): the kernel cannot account for it, since it replaces vLLM's single
`silu_and_mul_with_clamp` launch one for one (43 layers, a few rows each: tens of
microseconds at most against 0.43 ms), while the experts a step routes to follow the
text. Text-dependent throughput moved both ways (JSON +5.0%, c8 +2.8%, c1 distinct
-2.1%); the sixteen-prompt acceptance sweep
([precision-acceptance](../2026-10-11-precision-acceptance/README.md)) pairs the arms
over many texts. Mixed-traffic short TTFT rose 6.7% in this arm and 6.8% in the closing
r6d arm, so it is drift, not the change. Prefill is above both r6d arms.
