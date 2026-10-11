# All-reduces past the one-shot limit, in rank order (r6d)

Owner, 2026-10-11: correctness first (temperature-0 determinism, native-weight
computation), and speed never below B12X. The native-precision audit of r6d against
DeepSeek's reference found one determinism gap in production.

## The gap

Under prefill sequence parallelism, CED's decoder layers (20-39) run on the compacted
rows: every decoding request's rows plus at most 128 rows of the prefill chunk
(SWA Bounded Replay, DeepSeek's documented design). They run without sequence
parallelism and all-reduce their outputs. sparknet's one-shot all-reduce takes
messages up to its 2 MiB dispatch limit (204 rows of 5120 BF16). Past it,
`CudaCommunicator.all_reduce` falls back to NCCL's ring, which adds in another order
and rounds to BF16 at every hop. A step of 13 or more verifying streams plus a
prefill chunk (for example 15 x 6 + 128 = 218 rows) therefore reduces its rows with
other arithmetic than a smaller step, and a row's bits depend on the batch.
The r6d checks never reached that size: eight streams plus a chunk is 176 rows.

## The change (vLLM branch `r6d-allreduce`, on r6d's tree)

- `50892174e` (module): `rank_order_sum`'s torch path no longer adds into its first
  input (the fused-sum experiment's fix).
- `95fa3ac65` (module): `all_reduce_rank_order`. The rows, zero-padded to a multiple
  of the world size, go through the rank-order reduce-scatter (C3), then an
  all-gather, which only moves bits. Every row is the FP32 rank-order sum with one
  rounding, as in the one-shot all-reduce.
- `86b35b41e` (switch): under the TileLang family on the RoCE policy, all-reduces
  past the one-shot limit take it instead of NCCL's ring.

## Window 1 ([w1.json](w1.json))

- Kernel bundle: the collectives tests (host-simulated exchange and all-reduce at 2-4
  ranks, the GPU sum) in the r6d image.
- Fabric ([fabric_all_reduce.py](fabric_all_reduce.py)): at 205-8192 rows on the
  ring, the rank-order all-reduce and the communicator's all_reduce (with the
  switch) against the one-shot all-reduce of the same rows in 204-row chunks, bit for
  bit; NCCL's ring timed beside them and its differing elements counted.
- Serving: r6d against the arm, bracketed, the lean screen. `token_determinism.py
  --wide` adds `mixed15`: fifteen streams with a long prompt arriving mid-decode,
  each compared with its prompt served alone.

Acceptance: the fabric results equal the one-shot arithmetic bit for bit; every
temperature-0 scenario identical, references the same as r6d's (no step of those
reaches the new path); decode, prefill and mixed traffic level.

## Status

Window 1 (2026-10-11, 07:10-07:31 UTC): the bundle passes. Fabric, slowest rank's
median per call over 20 (rank-order elements differing from the one-shot arithmetic,
direct / through the communicator; NCCL's ring elements differing):

| Rows | MiB | differ (direct / communicator) | NCCL differs | NCCL | rank order |
| --- | ---: | --- | ---: | ---: | ---: |
| 224 | 2.19 | 0 / 0 | 377,052 | 743 us | 903 us (1.21x) |
| 300 | 2.93 | 0 / 0 | 505,839 | 791 us | 1,038 us (1.31x) |
| 512 | 5.00 | 0 / 0 | 862,531 | 979 us | 1,339 us (1.37x) |
| 1024 | 10.0 | 0 / 0 | 1,723,983 | 1,513 us | 1,854 us (1.23x) |
| 4096 | 40.0 | 0 / 0 | 6,899,246 | 3,814 us | 4,345 us (1.14x) |
| 8192 | 80.0 | 0 / 0 | 13,793,183 | 6,387 us | 7,397 us (1.16x) |

Serving (r6d / arm / r6d bracket): r6d fails `mixed15` in both runs the same way, 13
of 16 requests with other tokens and all 16 with other logprobs; the arm has 0 token
and 1 logprob difference (prompt 11, from token 7, when the long prompt's first chunk
joins the step). Every other scenario is identical and the references have r6d's
bits in all three runs. Speed is level: c1 steps within 0.4%, distinct c1/c8 within
0.4%, mixed traffic within the bracket; cold prefill reads 1.0-1.2% lower than the
first r6d run and 0.6% than the second, but a single prefill compacts its decoder
steps to 128 rows and never takes the new path.

The remaining difference occurs once the long prompt's chunk joins fifteen decoding
streams. The candidates the eight-stream mixed check (176 rows, identical) did not
cross are few; the source audit found no other row-count switch between 177 and 218
rows. Window 2 ([w2.json](w2.json), no speed bench) bisects: the arm with the 15-stream
check twice (does the one difference repeat in place?) and the arm with
`SPARK3_ENGRAM_ASYNC=0` (`allreduce-sync.json`: the Engram rows read before the
forward instead of beside it; the rows are the same, only their timing changes).

Window 2 (2026-10-11, 07:41-07:55 UTC): the arm's `mixed15` twice and the arm with
synchronous Engram reads give the same single difference every time: prompt 11,
logprobs only, from token 7, as in window 1. It is deterministic, not a race, and not
the asynchronous Engram reader; every other scenario stays identical and the
references keep r6d's bits. Something computes a row with other arithmetic in some
step shape that fifteen streams reach and eight do not.

Window 3 ([w3.json](w3.json)) runs `mixed15` on the arm without CUDA graphs
(`allreduce-eager.json`, `--enforce-eager`): graph replays pad decode steps to the
capture sizes (90 rows to 96) and eager steps do not. If the difference goes away the
cause is the graphs or their padding. Its second job localizes it if it stays:
`allreduce-trace.json` (eager, with the debug row trace of vLLM `r6d-row-trace`
`8fd502d47`, `VLLM_DS41_ROW_TRACE_DIR=/cache/row-trace`) and
[row_trace_session.py](row_trace_session.py), which traces the rows feeding prompt
11's tokens 4-9 alone and in mixed15 and reports the first layer whose digests differ.

Window 3 (2026-10-11, 08:58 UTC) did not boot: vLLM refuses DSpark's adaptive
verification without full CUDA graphs ("Adaptive verification requires full CUDA
graphs"), so the eager arm stopped at configuration on every rank and the window closed;
production was restored at 09:00. The arm's code never ran.

Window 4 ([w4.json](w4.json)) splits the bisect so each arm changes one thing:
`allreduce-fixed.json` keeps the CUDA graphs and turns adaptive verification off (every
draft token verified, as the eager path must), and `allreduce-eager.json` is that arm
with `--enforce-eager`. Both run `mixed15`. If "fixed" loses the difference, the variable
verification length is involved; if "fixed" keeps it and "eager" loses it, the graphs or
their padding are. The trace (`allreduce-trace.json`, now also without adaptive
verification) runs last and localizes the difference if eager keeps it.

Window 4 (2026-10-11, 13:21-13:33 UTC):
- **`allreduce-fixed`** (CUDA graphs, adaptive verification off): mixed15 0/16 and
  every other scenario identical. Its references equal r6d's (`6659a69ad547` /
  `15c551803076`).
- **`allreduce-eager`** (no graphs, adaptive verification off): the same, 0/16.

So the one difference left with the rank-order all-reduce needs DSpark's adaptive
verification. Adaptive verification trims each request's verified drafts on the GPU from
batch-level cost curves, so the verify step's shape follows the batch. Graphs and their
padding are not involved.

The row trace (eager, adaptive verification off) reported digest differences in three
rows between the prompt alone and mixed15, at layers 0-2. The tokens and logprobs are
identical, and one of those rows ran in a step of the same size, so the trace's row
matching (verify rows against draft rows) needs review before those differences mean
anything.

Next: localize the adaptive-verification difference. That needs a trace that records
inside the graphs, because adaptive verification requires them, or a source audit of
the kernels that see the trimmed batch's rows.
