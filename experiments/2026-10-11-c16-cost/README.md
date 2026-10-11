# Code c16: r6f's step cost against r6d's

Same-session benches (2026-10-11): r6f is 8.3% below r6d at code c16 (489 against 533
tok/s). It verifies 5% more rows a step (48.7 against 46.2) but its steps are 21% longer
(81.5 against 67.5 ms): 1.67 against 1.46 ms a verified row. At prose c16 (about 38 rows
a step) r6f is 10% faster a row. So r6f's step cost climbs past about 48 rows.

Both arms run on pinned DSpark cost curves (`SPARK3_DSPARK_COST_DIR`, r6's of
2026-10-05). Adaptive verification sizes its verification from those curves, so it
cannot see a cost r6f adds at large steps.

## Window 1 ([w1.json](w1.json))

r6d and r6f booted without pinned curves and with real-row profiling
(`SPARK3_DSPARK_PROFILE_TOKENS=random`). Each boot measures and logs its verify and
draft step cost at fixed token counts; the table puts them side by side. Where r6f's
verify cost leaves r6d's names the step size, and a torch profile there names the
kernel.

Window 1 (17:24-17:28 UTC) wrote no curves: a boot without a cost directory profiles
but neither logs nor saves the curves. Window 2 ([w2.json](w2.json)) gives each arm an
empty cost directory (`/cache/kkref/dspark-costs/c16-<arm>-20261011`). The boot then
profiles and pins its curves there: draft, verify, and verify per request count.
