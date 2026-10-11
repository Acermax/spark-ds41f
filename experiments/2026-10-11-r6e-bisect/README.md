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

### Window 1 results (2026-10-11, 14:28-14:57 UTC)

| arm | determinism | references (prompts alone / long prompt) |
|---|---|---|
| p1-allreduce | identical | `6659a69ad547` / `15c551803076` (r6d's) |
| p2-moe | identical | `e65c210d56ec` / `2216f9fddec8` |
| p3-head | identical | `61d4ba03df91` / `7d30f9fb21b9` |
| p4-precision | prompt 4 in mixed and mixed15 | `72f42af70fee` / `e60feca808f6` |
| p5-b3 | the same | the same as p4 (B3 keeps the bits) |
| p6-b4 | the same | `7ae87256d3ea` / `b8b35451dcc4` (r6e's: T1 keeps the bits) |

Each failing arm differs the same way: logprobs from token 4, tokens from token 11. The
first failing prefix adds two changes:
- the FP32 mHC projection: fn split into three exact TF32 parts, three GEMMs
  accumulating in FP32;
- the shared expert's SwiGLU at the reference's precision: an elementwise kernel.

Read alone, both are row-local: every row gets the same K order and the same ops
whatever its step. More likely they changed the numbers enough to expose an older
batch dependence. The all-reduce bisect's arm showed a different prompt (11) under
different numbers, and its trace found digest differences in layers 0-2.
- Prompt 11's row 21 differed after layer 1 inside an 8186-row mixed step.
- Its row 23 differed after layer 0 in a step of the same size as alone. That points to
  state an earlier step left behind (KV cache, compressor ring) rather than the step's
  own arithmetic.

## Window 2 ([w2.json](w2.json))

One arm, `r6e-trace`: the full candidate without adaptive verification, eager, with the
debug row trace (vLLM `r6e-trace` `362b78b80`, VLLM_DS41_ROW_TRACE_DIR).
1. `token_determinism.py --wide`: does the difference survive eager?
2. [row_trace_session.py](row_trace_session.py) on prompt 4. It traces the prompt's last
   six rows and the rows feeding generated tokens 1-6, alone and in mixed15, and reports
   each row's first differing layer. Changes from the all-reduce bisect's session:
   - the prompt's own rows, via the server's tokenizer;
   - generated rows from the first;
   - records tagged by rank;
   - rows that another prompt's row also matches in a step marked ambiguous.

### Window 2 results (2026-10-11, 15:00-15:07 UTC)

- **Eager:** the difference survives without CUDA graphs. Prompt 4 differs in mixed and
  mixed15 (logprobs from token 4, tokens from token 11), with r6e's references.
- **Trace:** the session's mixed15 phase took the server down. The trace indexed a
  sequence-parallel layer's unreduced parts (`RankOrderParts`) as a tensor and raised
  on rank 2. The other ranks then timed out in sparknet's relay (5 s). vLLM `a8b3ef9aa`
  digests such parts by their rank-order sum, the value a step without sequence
  parallelism holds.

Window 3 ([w3.json](w3.json)) reruns the trace session alone on the fixed trace.
