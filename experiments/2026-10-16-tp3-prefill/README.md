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

## Results (2026-10-16, 06:40-06:47 UTC)

The 16K prefill (16,388 tokens) takes 3.963 s on r6n, 3.901 s on r6h and 3.977 s on r6n again
(unprofiled warm passes 3.958 / 3.906 / 3.956 s): r6n is about 1.4% slower at this size, well
beyond the two r6n boots' difference.

r6h's TileLang kernels predate their names (all `main_kernel`), so
[match_grids.py](match_grids.py) pairs them with r6n's by grid, block and call count, and
[wo_totals.py](wo_totals.py) totals the paths that differ (ms per rank over the request):

| rank | build | span | WO-A and inverse RoPE | WO-B (MXFP8) | Engram wait kernel | NCCL | everything else |
|---|---|---:|---:|---:|---:|---:|---:|
| dgx1 | r6h | 3869.6 | 270.0 (B12X, FP8) | (in B12X's) | (not a kernel) | 614.0 | 2927.5 |
| dgx1 | r6n | 3931.2 | 204.5 | 124.0 | 53.0 | 642.0 | 2927.6 |
| dgx2 | r6h | 3866.5 | 269.3 | | | 569.7 | 2936.9 |
| dgx2 | r6n | 3928.2 | 211.3 | 126.4 | 86.2 | 586.6 | 2933.7 |
| dgx3 | r6h | 3866.5 | 270.5 | | | 544.2 | 2924.3 |
| dgx3 | r6n | 3925.2 | 216.3 | 124.8 | 119.2 | 553.8 | 2929.6 |

- **The WO projection is the loss.** Everything outside WO, the Engram wait and NCCL is level
  (within 5 ms of 2,930 on every rank). r6n's WO takes 329-341 ms per rank against r6h's 270
  (B12X's fused FP8 WO with its quantize and inverse-RoPE kernels): 59-71 ms more, the whole
  59-62 ms span difference. r6n against r6n again moves 4 ms.
- WO-A runs on BF16 activations (the reference's precision) where B12X quantized them to
  FP8, and r6k's TileLang tiles were swept on TP4's per-rank shapes. On TP4, r6k's prefill
  was 0.5-0.8% faster than r6h, so the TP3 shapes are the gap to work on: sweep WO-A and WO-B
  tiles for TP3's per-rank shapes (ceiling: about 60 ms of a 16K prefill, 1.5%).
- NCCL grows 10-28 ms per rank, consistent with waiting on slower peers.
- **Engram wait:** r6n's one-thread wait kernel spins 53 / 86 / 119 ms on dgx1 / dgx2 / dgx3
  (1.4-3% of the request) for the rows of the 16K prompt. r6h waits too, as a stream wait that
  is not a kernel, so the profile cannot compare them; the span difference is fully explained
  by WO. It is the larger lever: prefill read-ahead (TODO, Engram after r6n) would take
  the wait off the critical path, and dgx3, the slowest reader, paces every rank.
