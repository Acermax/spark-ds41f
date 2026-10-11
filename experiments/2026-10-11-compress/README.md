# Port B4: CSA compression in TileLang (r6d)

The TileLang migration's B4 (see the
[inventory](../2026-10-09-tilelang-migration/README.md)). Each of the 38 compressor
layers (20 at ratio 1, 18 at ratio 2) emits the 512-wide latents the indexer and the
compressed KV cache consume, each step:
- **Ratio 1:** the RMS norm of every token.
- **Ratio 2:** each odd position pools its pair with a per-channel softmax of the
  gates, rounds to BF16, then takes the norm.

r6d runs four launches per ratio-2 compressor:
- vLLM's Triton carry load;
- vLLM's Triton ring save, one program per token of the batch capacity;
- B12X's `mla_compress`, a 128-thread block per token of the capacity, 8192 at TP4,
  each writing its latent row even past the live tokens;
- B12X's pending-ring commit, whose ring vLLM reloads from the cache every step.

A September decode profile had `mla_compress` at about 11.5 us a call, about 0.44 ms
of a 31.5 ms c1 step.

## The change (vLLM branch `r6d-compress`, on r6d's tree)

- `a3b21cdda` (module): `tilelang/compress.py`.
  - `gather_partial_state` is vLLM's carry load.
  - `compress` runs over the step's rows only: one warp a token, 16 values a lane,
    a warp-shuffle sum of squares, and library `exp` and `rsqrt` as DeepSeek's
    reference computes them (B12X used fast approximations).
  - It also saves each request's latest ring positions, so a slot holds its latest
    position. vLLM's Triton save let every position of a long prefill chunk race for
    its slot. Chunk alignment keeps those slots unread today, so this changes no
    output.
- `49e598640` (switch): the compressor runs the two TileLang launches under the
  TileLang family; they compile at preparation. B12X's preparation still allocates
  the compressor's buffers (migration step 7).

The latents are not bit-equal to B12X's: the reduction order and library `exp` and
`rsqrt` differ, within one BF16 ulp of DeepSeek's arithmetic. Outputs and references
may move from r6d's.

## Window 1 ([w1.json](w1.json))

1. Kernel bundle.
   - The tests check the latents against DeepSeek's arithmetic, and the carries and
     short-request ring saves bit for bit against vLLM's Triton kernels. They also
     check batch independence and graph replays.
   - Then [bench_compress.py](bench_compress.py), at ratio 1 and 2, decode steps and
     prefill chunks, warm and cold:
     - emitted flags and slots equal to B12X's;
     - error against an FP64 recompute no worse than B12X's;
     - time per step against B12X's whole path;
     - replay repeatability.
2. Lean screen bracketed by r6d, with `--wide` determinism and the sixteen-prompt
   acceptance sweep, since the text may move.

Acceptance:
- temperature-0 identical within the arm;
- the bench faster than B12X's path at every step (the owner's bar);
- step time at or above r6d, and paired acceptance level.
