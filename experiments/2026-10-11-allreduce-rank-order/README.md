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
