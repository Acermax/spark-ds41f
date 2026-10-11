# r6g: r6f's decode cost back, at r6f's bits

The [fixed-row profile](../2026-10-12-fixed-row-profile/README.md) put r6f's step 5.6%
above r6d's at 96 verify rows. Two fixes keep every bit:

1. **The shared expert folded at decode** (this window). The fold (vLLM 0109) adds the
   shared expert's output in the MoE combine and rounds once, but only when the shared
   expert runs first on the same stream. At decode (up to 256 tokens) it runs on the aux
   stream beside the routed experts, so the combine wrote an FP32 partial (the grid
   (40, 6) kernel, 30.7 to 43.2 µs a call) and a separate pass added and rounded at
   every layer (0.29 ms a step). vLLM `r6g-fold-overlap` `df4f1d98f` launches the shared
   expert on the aux stream before the routed experts and has the plan wait for it just
   before its combine. The arithmetic is the fold's: the same bits.
2. **Greedy verification without the logits gather** (next): the target's FP32 logits
   all-gather (2.1 ms a step on NCCL) and full-vocabulary pass (0.4 ms).

## Window 1 ([w1.json](w1.json))

- Measure job, r6f / fold / r6f again on the promoted image, each with real-row cost
  curves (empty cost directory, `SPARK3_DSPARK_PROFILE_TOKENS=random`):
  `token_determinism.py --wide` (the fold's references must equal r6f's,
  `7ae87256d3ea` / `b8b35451dcc4`), and sixteen and eight different prompts at once.
- Profile job, r6f and fold with adaptive verification off (96 rows a step), the
  bench's code prompt on sixteen streams under the torch profiler.

The overlay mounts the four changed files over the r6f image.
