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

### Gate window 2 results (2026-10-11, 13:26-13:52 UTC)

r6d / r6e / r6d bracket, lean screen:
- **Decode, one prompt each:** prose -8.4% tok/s with acceptance 1.257 to 1.056; JSON
  step +3.4% with acceptance 3.000 to 3.127, tok/s level. Both follow the changed text,
  not the kernels.
  - Paired over the sixteen-prompt acceptance sweep, r6e against r6d is +0.2% ±2.3%
    tok/s and +1.2% ±4.8% acceptance. The bracket arm is +0.0% ±0.2%.
- **Distinct prompts one at a time:** +2.5%.
- **Cold prefill:** +1.0% at 1024 tokens, +1.6% at 16384.
- **Eight concurrent prompts:** +0.3%.
- **Mixed traffic:** short TTFT -4.4%. Long TTFT read +2.7% (three rounds, 925-942
  against r6d's 884-932 ms); adaptive window 1 measured 917 and 916 ms for r6e.
- **Determinism:** every scenario identical except `mixed15`, 1/1 (r6d: 13/16). The
  references differ from r6d's, as the precision changes must.

### Adaptive verification window 1 results (2026-10-11, 14:00-14:23 UTC)

r6e / r6e-fixed (adaptive verification off) / r6e bracket:
- **Cost of turning it off:** decode -7.9% ±2.4% tok/s (paired acceptance sweep),
  -4.6% on distinct prompts, -23.9% at eight concurrent prompts, and mixed traffic 26%
  fewer chunks a second with short TTFT +29%.
- **Determinism:** r6e-fixed still differs, in `mixed` and `mixed15`: prompt 4, logprobs
  from token 4, tokens from token 11. r6e and its bracket differ the same way in
  `mixed15` only, as in gate window 2.
- **References:** identical across the three arms, as greedy verification of every
  draft must be.

Adaptive verification stays on. The remaining difference is reproducible and has
another cause. The [bisect](../2026-10-11-r6e-bisect/README.md) finds the piece.
