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
