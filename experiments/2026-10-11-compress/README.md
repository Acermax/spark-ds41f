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

- `c1e10dce3` (module): `tilelang/compress.py`.
  - `gather_partial_state` is vLLM's carry load.
  - `compress` runs over the step's rows only: one warp a token, 16 values a lane,
    a warp-shuffle sum of squares, and library `exp` and `rsqrt` as DeepSeek's
    reference computes them (B12X used fast approximations).
  - It also saves each live token's ring state for the rows the metadata gives a
    ring slot. Those are a request's last ring-size rows; earlier rows would share
    slots. The kernel keeps that bound itself as well.
- `fa7be7d63` (switch): the compressor runs the two TileLang launches under the
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

### Window 1 (2026-10-11, 12:05-12:16 UTC)

**Kernel bundle:** the bench passed outright.
- **Speed:** faster than B12X's whole path at every step and both ratios, with the
  same FP64 error at every size.
  - Ratio 1: 17.1-17.3 against 1.7-2.1 us warm at decode, about 10x; 31.5 against
    24.7 us at 8192 prefill rows.
  - Ratio 2: 31.6-36.3 against 6.4-6.7 us at decode, about 5x; 328 against 191 us at
    8192 rows.
- **Expected serving gain:** about 0.8 ms of a 31.5 ms c1 step.
- **Tests:** two failed, both on the test's side.
  - The random block table could give two requests one ring block, so their saves
    raced.
  - One element was 2 BF16 ulps from the torch arithmetic. The library exp and the
    reduction order can each move an FP32 bit.
  - vLLM `0157ed22a` gives each request its own block and allows two ulps.

**Serving:** the arm did not boot.
- The KV allocator pads its pages, so the ring pool's rows are strided. The kernels
  declared it packed: "gather_partial_state input pool strides[0] violates packed ABI
  constraint".
- vLLM `e849609f9`, on top, takes the pool with its row stride, and the tests use a
  padded pool.

Window 2 ([w2.json](w2.json)) is window 1 again.
