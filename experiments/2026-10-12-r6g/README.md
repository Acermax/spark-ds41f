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

### Window 1 results (2026-10-11, 22:41-23:06 UTC)

| metric | r6f | fold | r6f again |
|---|---|---|---|
| determinism `--wide` | identical | identical | identical |
| references | `7ae87256d3ea` / `b8b35451dcc4` | the same | the same |
| sixteen different prompts at once, tok/s | 222.4 ±0.4% | **225.2** ±0.8% | 221.8 ±1.0% |
| eight different prompts at once | 177.6 ±1.0% | 178.7 ±1.3% | 177.7 ±1.6% |
| verify cost, 24-96 rows (real rows) | | -2.3% | |

The fold keeps r6f's bits and is 1.4% faster at sixteen prompts (against the two r6f
arms' mean); eight is level. The fixed-row profile shows the separate add-and-round
pass gone (281 µs a step to 0) and the sparknet all-reduce 12 µs a call faster. Under
the torch profiler the routed experts' grouped GEMMs ran 15-22% longer a call with the
fold and the profiled window was slower, which the unprofiled cost curves and
throughput do not show: the shared expert now starts on the aux stream after the gate
and shares SMs with the routed GEMMs. Open: whether a later aux launch (after the
gate/up GEMM) gains more.

The r6f arms read 222.4 and 221.8 at sixteen prompts against 218.0 in the release
session of 2026-10-11: sessions differ by about 2%, so the verdict against r6d comes
from one session with both.

## Window 2 ([w2.json](w2.json)): both fixes

r6f / r6g / r6f again. r6g mounts vLLM `r6g-fold-overlap` `bbd7157f0` over the r6f
image: the window 1 fold and greedy verification from vocabulary shards (an
all-greedy step whose logits nothing reads reduces each rank's shard to a
`(max, index)` pair, gathers the pairs and applies `rejection_sample`'s greedy rule;
`VLLM_GREEDY_VERIFICATION=0` keeps the gathered path).

- Kernel bundle first ([bundles/r6g](bundles/r6g)): the deferred shared addend against
  the direct one, and the shard argmax against `torch.argmax` (ties across shard
  boundaries, junk in the padding) and the greedy rule against `rejection_sample`.
- [token_determinism_text.py](token_determinism_text.py) twice per arm: with logprobs
  (the gathered path; r6g's references must be r6f's), and `--no-logprobs` (r6g's
  greedy path: every scenario identical, and the reference texts' digest equal to the
  logprobs run's).
- Sixteen and eight different prompts at once (no logprobs: the greedy path) and
  real-row cost curves.

## Window 3 ([w3.json](w3.json)): the greedy switch alone

Window 2's r6g read +0.8% at sixteen prompts against the r6f arms' mean (+1.2% at
eight), where window 1's fold alone read +1.4%: the greedy path's share is not visible
at production shapes. The same overlay with `VLLM_GREEDY_VERIFICATION` on, off and on
again, six samples of sixteen and eight different prompts, then a fixed-row profile of
both (adaptive verification off): with the switch on the logits all-gather and the
full-vocabulary passes must disappear.

### Window 3 result and window 5 ([w5.json](w5.json))

Window 3 measured no difference with the switch, and its profile showed why: with the
switch on, the stock path still ran (the NCCL logits all-gather, the logits-stats and
rejection kernels). A logged run (window 4) found the guard off: the served model is
the vision-language wrapper, which delegates `compute_logits_local` but has no
`lm_head` or `logits_processor` of its own. vLLM `fbaa04151` resolves both on the
language model and logs once whether the path is on; the overlay now carries it (with
`3d441bda9`'s warm-up). Window 2's greedy determinism run therefore exercised the
gathered path, and its gate is rerun here: window 5 repeats window 3 with both
determinism modes on every arm.
