# Fixed-row kernel profile: r6f against r6d

[c16-cost](../2026-10-11-c16-cost/README.md) window 5 put r6f's verify step 3.2% above
r6d's at 24-96 rows and level at 1-20, nearly all of it from the precision pieces (MoE one
rounding, FP32 head, FP32 mHC and SwiGLU). The c16 torch profile could not say which
kernels grow, because the two releases generate different text: acceptance, verified
rows and dead-row pruning all moved with it.

## Window 1 ([w1.json](w1.json))

Both releases with adaptive verification off, so every step verifies the anchor and all
five drafts of every request (dead-row pruning runs only under adaptive verification).
The shapes are then the same in both arms whatever the text: 96 rows a step at sixteen
streams (`code16`), 48 at eight (`c8`). Each arm runs under the torch profiler; the lab
collects every rank's trace and compares kernels per step against r6d with
`analyze_costs.py`.

- [r6d-fixed.json](r6d-fixed.json): r6d's TP4 profile, adaptive verification off, profiler on.
- [r6f-fixed.json](r6f-fixed.json): the promoted r6f profile, the same changes.
- [profile_c8.py](profile_c8.py): the c16 profile's workload script, reading `--case`,
  `--streams` and `--tokens`.
