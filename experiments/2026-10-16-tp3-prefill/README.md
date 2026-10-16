# TP3 prefill: where r6n's time goes against r6h

r6n's TP3 release (experiments/2026-10-15-r6n-tp3) read prefill 0.3-0.5% slower than r6h
at every size (32K-500K). r6n's own repeat moved as much, so the release counts it as level,
but the arms ran in time order (r6n, r6h, r6n again) and the repeat drifted down 0.5%, so a
small real cost of about 0.5% is plausible. TP4 shows no such loss (r6h to r6k +0.5-0.8%
at long prefill, r6k to r6n level), so the suspects are what TP3 changes:

1. **r6k's TileLang WO** (WO-A on BF16 activations, B12X used FP8), its tiles tuned on TP4's
   per-rank shapes; a TP3 rank holds a third of WO.
2. **Engram rows v2**: a TP3 rank reads 8 of the 24 heads' rows (TP4: 6); v2 was
   benchmarked against B12X only on TP4's quarter shard. The forward waits for the rows at
   layers 1 and 14 (r6n's one-thread `engram_rows_wait` kernel shows that wait on the GPU).

Not the cost curves: they size verification only.

## Method ([profile.json](profile.json))

The lab's profile job: each arm boots with vLLM's torch profiler
([r6n-tp3-prof.json](r6n-tp3-prof.json), [r6h-tp3-prof.json](r6h-tp3-prof.json)), and
[profile_prefill.py](profile_prefill.py) profiles one cold 16K prefill of servebench's
prefill document (the release suite's text), after an unprofiled warm prefill of it. Every
rank's trace is collected; with the cluster stopped, [summarize_kernels.py](summarize_kernels.py)
and [analyze_costs.py](analyze_costs.py) (`--total`) compare kernel time by family, rank by
rank, against r6n. Arms: r6n, r6h, r6n again: the second r6n bounds a family's
boot-to-boot noise.
