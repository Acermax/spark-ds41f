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

### Window 3 results (2026-10-11, 15:13-15:14 UTC)

Prompt 4 is 18 tokens; the trace covers rows 12-23.

| rows | first difference | step rows alone / mixed |
|---|---|---|
| 12-14 (prompt) | none | - |
| 15-17 (prompt) | layer 0, ambiguous in both phases | 18 / 250 |
| 18 | layer 0, ambiguous in mixed | 6 / 90 |
| 19-20 | none | - |
| 21 | **layer 1** | 6 / 6814 |
| 22-23 | layer 2 | 6 / 6814 |

All-reduce window 4's trace of prompt 11 showed the same unambiguous pattern: row 21
first differed at layer 1 inside an 8186-row mixed step, row 22 at layer 2.
- A decode row inside a large mixed (sequence-parallel prefill) step is identical
  through layer 0, then differs after layer 1.
- The next rows differ from layer 2 on, as their attention reads the first row's
  changed layer-2 KV.

Layers 0 and 1 are both sliding-window layers without compression (compress ratio 0).
Layer 1 alone adds Engram (`engram_layer_ids` = [1, 14]). So the batch dependence is in
the Engram path for a decode row in a large step:
- the hash of its n-gram;
- the staged table rows and their all-reduce;
- the `wkv` projection;
- the gate.

The gate kernel is row-local (one block per token and stream, a fixed launch). `wkv` is
the TileLang FP8 linear that every layer uses. The all-reduce has one nonzero
contributor per element.

The ambiguous rows had two matching records in one step and label. The drafter does
not record (it is another model class), so another row shares their position and
token there. Window 4 reports which rows.

## Window 4 ([w4.json](w4.json))

The trace session again, on vLLM `c2615923f`:
- lines carry the recording model;
- Engram layers record every token's hash ids, staged rows (after the all-reduce),
  projection, and this rank's gated rows (`engram1_hash`, `_rows`, `_kv`, `_out`);
- the session keys labels by model, lists each row's differing labels in order, and
  names the colliding rows of an ambiguous match.

### Window 4 results (2026-10-11, 15:28-15:30 UTC)

Row 21 (generated, in the 6814-row mixed step): every Engram stage of layer 1 is
identical (`engram0_hash`, `_rows`, `_kv`, `_out`), yet layer 1's output differs. The
difference arises after Engram, in layer 1's attention, mHC or MoE. Those same
operations left layer 0 identical. So the batch dependence is sparse and depends on the
values, not on the row:
- an ulp-level difference that matters only at a discrete boundary, such as top-k near
  a tie or an FP8 rounding edge;
- here it is crossed at layer 1, not at layer 0.

The ambiguous rows are now named. Prompt rows 15-17 collide with another prompt's
rows at the same positions in step 160. A step can hold two requests whose chat
templates put the same token at the same position.

One candidate stood out reading the code. Steps up to the gate's split rows route
through a fused gate router (the gate's split-K partials reduced and routed in one
kernel). Larger steps run the gate linear and then TileKernels' routing. The test
compares the two bit for bit at the same row count, not across row counts.

## Window 5 ([w5.json](w5.json))

The trace on vLLM `c54915432`. Each decoder layer records:
- the residual entering an Engram layer;
- the attention input and output;
- the FFN input.

The routers record every routed row's weights and expert ids under one label (`L<i>_route`)
whichever path routed it, and its gate logits under the path's label. The first
differing label inside layer 1 names the stage.

## Window 6 ([w6.json](w6.json), kernel only)

The router across step sizes. `test_gate_router_rows_do_not_depend_on_the_step`
(vLLM `r6e-router-check`) routes a 6814-row batch, then its first 1-2048 rows. It
requires every row's gate logits, expert ids and weights to equal the full batch's bit
for bit. Small row counts take the fused gate router, larger ones the gate's in-kernel
shards then TileKernels.

### Window 5 (2026-10-11, 16:01-16:03 UTC): the trace crashed on the drafter

The DSpark drafter's decoder layers are the same class. They recorded into the
target's trace with the target's matched row indices and indexed past their own five
rows (`IndexError`), which took the server down. vLLM `f9669d2c4` ends the trace with
the target model's forward and skips rows a tensor does not hold.

### Window 6 results (2026-10-11, 16:03 UTC)

The router tests passed (47), `test_gate_router_rows_do_not_depend_on_the_step`
included. At 1-2048 rows every row's gate logits, expert ids and weights equal the
6814-row batch's bit for bit, through both the fused gate router and the gate then
TileKernels. The routing is not the batch dependence.

## Window 7 ([w7.json](w7.json))

Window 5's trace on `f9669d2c4`.
