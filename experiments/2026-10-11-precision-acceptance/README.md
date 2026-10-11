# DSpark acceptance across texts for the precision arms

The precision fixes (FP32 head logits, FP32 shared-expert SwiGLU) bring temperature-0
outputs closer to DeepSeek's reference and so change the text. The lean screen's
single-stream points each decode one prompt, so their tok/s moves with that prompt's
new continuation: FP32-head window 1 had prose-c1 at -6.1% (accepted/draft 1.116
against 1.257) with GPU step time level, and JSON (acceptance at its 3.000 maximum in
every arm) level. One text cannot separate an acceptance shift from a different text.

[`scripts/acceptance_sweep.py`](../../scripts/acceptance_sweep.py) runs each of the
sixteen distinct-stream prompts alone (thinking off, temperature 0, 256 tokens, fresh
cache salt) and reads the server's speculative-decoding counters around each one.
[tables_arms.py](tables_arms.py) pairs the arms prompt by prompt against r6d.

## Window 1 ([w1.json](w1.json))

No bench: boot r6d, the FP32-head arm, the shared-SwiGLU arm and r6d again, each with
the sweep (about two and a half minutes per arm). Acceptance: the paired change in
accepted tokens per draft and in tok/s covers zero or is positive, and r6d against
itself stays within the interval. A loss that holds across the prompts would mean the
drafter disagrees more with the more precise target; the next step would then be the
drafter's own precision, never the target's.
