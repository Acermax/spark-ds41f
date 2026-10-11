# r6e candidate (TP4)

Everything validated since r6d, as one arm over the r6d image. It is built from vLLM
branch `r6e-draft2` (`1e9ac8c16`, 28 commits on `r6d-candidate`) by
[make_arms.py](make_arms.py), with `VLLM_DS41_HEAD_DTYPE=float32` for the FP32 head.

| piece | evidence |
|---|---|
| SP WO projection in two slices (overlap) | identical bits, +2% at 16K prefill (PR #50) |
| fused SP sum in mHC | same bits, about +1% prefill (PR #52, #54) |
| rank-order all-reduce past the one-shot limit | mixed15 13/16 to 1/16 differing (PR #58, #60); the last difference under bisect (all-reduce window 4) |
| MoE: one rounding of each rank's partial | deterministic, decode step level, paired acceptance level; 16K prefill -1.3% ([moe-one-rounding](../2026-10-11-moe-one-rounding/README.md)) |
| FP32 head logits | deterministic, step level, paired acceptance -1.0% ±1.9% (PR #66, acceptance sweep) |
| FP32 mHC projection | deterministic, cost under 2% (PR #64) |
| reference-precision shared SwiGLU | deterministic, prose step level, paired acceptance +0.2% ±3.0% (PR #67) |
| port B3, cache writers | faster than B12X at every size (migration windows 9-10) |
| port B4, compressor | 5-10x B12X's path per call; serving decode step -0.9 to -1.1%, acceptance sweep +1.1% ±0.1% ([compress](../2026-10-11-compress/README.md)) |
| port T1, sparse-MLA metadata | exact; 0.63-0.86x Triton at decode and prefill, builder 0.83-0.86x ([metadata](../2026-10-11-metadata/README.md)) |

Left out until they meet the B12X floor:
- **WO-A on BF16 activations (port B1):** decode 1.2-1.4x B12X. The next step is split-K.
- **Port B10, Engram lookup:** about 0.5 us slower than B12X a call at 1-32 tokens.

The first draft (`r6e-draft`, with MoE FP32 partials, WO-BF16 and an FP32-parts commit)
is superseded.

## Before the gate

- WO window 2 and MoE FP32 window 1 pass. Their pieces leave the stack if they fail.
- The new kernels compile: the mHC FP32-parts `post_pre` and the FP32 rank-order sum.
  Check them in a CPU-only container first.
- The remaining mixed15 difference is localized (all-reduce window 4).

## Gate window 2 ([w2.json](w2.json))

As below, without the fabric job, and with the acceptance sweep. Window 1 is the first draft's spec.

## Gate window 1 ([w1.json](w1.json))

1. Kernel bundle: every DS4.1 TileLang test the stack touches, in the r6d image.
2. Fabric: the MoE reduction, FP32 against BF16, with bits.
3. Lean screen bracketed by r6d:
   - `token_determinism.py --wide` (mixed15);
   - distinct prompts;
   - mixed traffic;
   - the sixteen-prompt acceptance sweep.

Acceptance:
- temperature-0 identical within the arm in every scenario, mixed15 included;
- decode, prefill and TTFT at or above r6d (paired acceptance for text-dependent
  throughput), and every metric above B12X's;
- then the promotion bench at the recipe limits (TP4, 1M context, 16 streams) and the
  owner's decision.

## Adaptive verification window 1 ([w3.json](w3.json))

The candidate with and without DSpark's adaptive verification (`r6e-fixed.json`: every
draft token verified), bracketed by the candidate. All-reduce window 4 found that the
last mixed15 difference needs adaptive verification on.
- **If the fixed arm is fully deterministic:** its speed cost decides between turning
  adaptive verification off and localizing the difference with a graph-safe trace.
- **If it is not:** the difference has another cause.
