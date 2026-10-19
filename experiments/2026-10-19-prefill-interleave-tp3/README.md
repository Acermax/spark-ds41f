# TP3 prefill interleaving with the CED skip (prefill follow-ups, item 6)

While a long prompt prefills, every engine step carries one of its chunks and the other
streams wait for it: with the skip at 4,096 their step is 0.86 s (median) instead of
0.10 s, about 30 s for a 131K prompt (experiments/2026-10-19-prefill-budget-tp3).
`--prefill-schedule-interval N` schedules that prompt's in-progress chunk only every N-th
step while other requests decode, so the steps between are decode-only. It counts steps,
not seconds (no timing in the decision; `--prefill-compute-share` meters measured step
time, so it is not tried). Chunk boundaries do not move, so aligned chunks and the skip's
successor check are unaffected, and a prefill with nobody decoding runs every step.
Expected: the other streams' median gap falls toward (step + (N-1) x 0.1 s) / N; the worst
wait does not change (one step still carries a chunk); the prompt's prefill under load
slows by about (N-1) x 0.1 s a chunk.

Arms, the gates' skip overlays on production r6x TP3 at 1M: intervals 1, 2 and 3 at the
budgets 4,096 and 8,192, then 4,096 at interval 1 again (the repeat).
- `interleave.json`: `suite.json`, the stall of seven decoding streams during a cold 131K
  prefill and that prefill's rate under them, three repeats per arm.
- `determinism.json`: outputs at intervals 2 (4,096) and 3 (8,192), 700- and 2,200-line
  long prompts, must equal the skip's (62313cb5 / 9742ac2c and 62313cb5 / f6171075):
  interleaving moves decode rows between steps, and the mixed scenarios put them beside
  the long prompt's chunks.

No promotion or production change.
