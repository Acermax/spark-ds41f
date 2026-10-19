# TP3 long-prefill budget with the CED skip (prefill follow-ups, item 5)

Each shape chooses its own `max_num_batched_tokens`; this sweep is tp3-triangle's, with
the skip (experiments/2026-10-19-ced-gates, both arms bit-identical to r6x; the skip +11%
on release-tp3 prefill at 4,096). A TP3 rank holds a third of the model, so its memory
headroom differs from TP4's; tp4-ring's sweep waits for the ring cabling.

Arms, the gates' skip overlays on production r6x TP3 at 1M, differing only in the budget
and its threshold (budget - 8 x 6): `skip-4096-tp3.json` (production's 4,096 / 4,048),
`skip-6144-tp3.json` (6,144 / 6,096), `skip-8192-tp3.json` (8,192 / 8,144). Without the
skip, 8,192 already booted past dgx1's startup guard and kept it 1.5 GiB above its running
guard through a 1M prefill (experiments/2026-10-19-r6x-tp3-1m), so no in-between arm is
needed for memory; 6,144 maps the trade between the two.

`ports-4096-tp3.json` is the gates' ports arm (no skip) at 4,096, the same-session
reference for the skip at 1M and for its effect on the stall.

`sweep.json` measures the skip at 4,096, the ports at 4,096, the skip at 6,144 and 8,192,
and the skip at 4,096 again (the repeat), with `suite.json`:
- release-tp3's prefill sizes, 32K to 500K, three repeats;
- `servebench stall`: what seven decoding streams wait while a cold 131K prompt
  prefills (TP3 serves eight sequences), three repeats: the worst wait (the longest step
  carrying a prefill chunk), the median and p99 step during the prefill, the baseline;
- three cold 1M prefills (TP3 serves 1M since r6x), during which the run's memory watch
  records dgx1's lowest MemAvailable against its 3 GiB guard.

Gate for a larger budget: dgx1 above its guard at boot and through the 1M prefill, and a
stall the owner accepts for the prefill it buys. tests/test_recipe.py pins each shape's
budget, so a change to one cannot move another. No promotion or production change.
