# r6e bisect: the difference without adaptive verification

r6e adaptive window 1 ([r6e-candidate](../2026-10-11-r6e-candidate/README.md)) ran the
candidate without DSpark's adaptive verification (`r6e-fixed.json`).
- **Speed:** turning adaptive verification off cost 7.9% ±2.4% decode tok/s, paired
  over the sixteen-prompt acceptance sweep.
- **Determinism:** it was not fully deterministic. Prompt 4 differed in `mixed` and
  `mixed15`, with logprobs from token 4 and tokens from token 11.

The candidate with adaptive verification differs the same way in `mixed15`, in gate
window 2 and in adaptive window 1. The difference is reproducible: a row computed with
other arithmetic in some step shape, not a race. It is not the all-reduce arm's
adaptive-verification difference (prompt 11, gone without adaptive verification), so
some piece r6e adds over that arm causes it.

## Window 1 ([w1.json](w1.json))

[make_arms.py](make_arms.py) builds six arms over the r6d image. Each mounts vLLM
branch `r6e-draft2` up to one more piece, and every arm runs without adaptive
verification. The head stays BF16 (r6d's `VLLM_DS41_HEAD_DTYPE=model`) until the
FP32-head commit, as in the stack.

| arm | prefix | adds |
|---|---|---|
| p1-allreduce | `a0d2d1fd1` | SP WO slices, fused SP sum, rank-order all-reduce |
| p2-moe | `14d3a5dbe` | MoE: one rounding of each rank's partial |
| p3-head | `3737e4b72` | FP32 head logits |
| p4-precision | `bfeb18c37` | FP32 mHC projection, reference-precision SwiGLU |
| p5-b3 | `37c727038` | port B3, cache writers |
| p6-b4 | `26f1b3055` | port B4, compressor |

The full stack (p6 plus port T1, the sparse-MLA metadata) is `r6e-fixed`, measured
twice already. Each arm runs only `token_determinism.py --wide` (no bench). The first
prefix that differs carries the cause; its piece's kernels get a cross-bucket bit
check (a row's bits at every row count against the full batch).
