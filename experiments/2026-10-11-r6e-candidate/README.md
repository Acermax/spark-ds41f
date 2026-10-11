# r6e candidate (TP4)

Everything validated or pending since r6d, as one arm over the r6d image. It is
built from vLLM branch `r6e-draft` (`d0815e76b`, 23 commits on `r6d-candidate`) by
[make_arms.py](make_arms.py), with `VLLM_DS41_HEAD_DTYPE=float32` for the FP32 head.

| piece | vLLM commits | evidence so far |
|---|---|---|
| SP WO projection in two slices (overlap) | `0c2a3add0`, `2eafb9f5e` | identical bits, +2% at 16K prefill (PR #50) |
| fused SP sum in mHC | `ecb6a9f31`, `3afd03827`, `1a63ac536`, `1c20bb4c8` | same bits, about +1% prefill (PR #52, #54) |
| rank-order all-reduce past the one-shot limit | `aa18d44cb`, `51a55c62a` | mixed15 13/16 to 0/1 differing (PR #58, #60); one logprob difference under bisect (window 4) |
| MoE FP32 partials | `5d7272a1b`, `00f86363b` | window 1 pending ([moe-fp32](../2026-10-11-moe-fp32/README.md)) |
| FP32 head logits | `ff30cad93` + `VLLM_DS41_HEAD_DTYPE=float32` | deterministic, step time level (PR #66) |
| FP32 mHC projection | `e62a7a8ee` | deterministic, cost under 2% (PR #64) |
| reference-precision shared SwiGLU | `ffb913aa3`, `9d0356c21` | deterministic, prose step level (PR #67) |
| BF16 WO-A, port B1 | `5d6f63aaa`, `bff18b5ad`, `05679a7d7` | window 2 pending ([wo-bf16](../2026-10-11-wo-bf16/README.md)) |
| port B3, cache writers | `0a1525bdb`, `53652ce13`, `bfcdfe0b3`, `23a86192f`, `92ff41cff` | faster than B12X at every size (migration windows 9-10) |
| FP32 MoE parts reach mHC unrounded | `d0815e76b` | new; see below |

Two textual conflicts were resolved in the stack: `collectives.py` exports and tests,
and `kernels.py` flags. One semantic interaction needed a commit. With MoE FP32
partials, the fused SP sum exchanges FP32 parts; the MoE layer then rounded each
rank's part to BF16 before mHC added them. The non-SP path rounds once after the FP32
sum, so a row's bits would have depended on whether its step ran sequence-parallel.
`d0815e76b` changes three things:
- the MoE layer leaves unreduced parts unrounded;
- mHC's `post_pre` takes FP32 parts, adds them in FP32 in rank order and rounds once;
- `RankOrderParts.sum()` rounds FP32 parts once to BF16.

## Before the gate

- WO window 2 and MoE FP32 window 1 pass. Their pieces leave the stack if they fail.
- The new kernels compile: the mHC FP32-parts `post_pre` and the FP32 rank-order sum.
  Check them in a CPU-only container first.
- The remaining mixed15 difference is localized (all-reduce window 4).

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
