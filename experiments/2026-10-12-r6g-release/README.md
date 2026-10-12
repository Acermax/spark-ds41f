# r6g against r6d: the release comparison on TP4

The [release protocol](../../docs/lab.md#release-comparison) (lab profile `release`), with
r6d run again after r6g: the r6g windows measured up to 2.3% between two arms of one
configuration twenty minutes apart, so r6d before and after bounds the drift.

- r6d: `experiments/2026-10-10-r6d-deterministic/r6d-tp4.json` (the previous release).
- r6g: `experiments/2026-10-12-r6g/r6g-tp4.json`, the baked `-r6g` image
  ([r6g](../2026-10-12-r6g/README.md): r6f with the aux-stream MoE fold and greedy
  verification from vocabulary shards, both bit-identical to r6f).

## Results (2026-10-12, 00:54-02:45 UTC, the ring)

`scripts/release_table.py r6g-release-tp4 r6d r6g r6d-again`. r6d run again after r6g
moved at most 0.3% on every decode and prefill metric (interval-level), so the session
did not drift.

| metric | r6d | r6g | change | r6d again |
| --- | ---: | ---: | --- | --- |
| temperature 0, fifteen streams + long prompt | 12 of 16 differing | **0** | | 13 of 16 |
| references | `6659a69ad547` / `15c551803076` | `7ae87256d3ea` / `b8b35451dcc4` (r6f's) | | r6d's |
| quality | 5/5 | 5/5 | | 5/5 |
| paired sweep, 1 stream (16 prompts) | | | +1.3% ±2.3% | -0.1% ±0.2% |
| 4 different prompts at once, tok/s | 132.0 | 135.1 | +2.3% ±2.8% | +0.3% |
| 8 different prompts at once | 176.4 | 178.1 | +0.9% ±1.1% | +0.0% |
| 16 different prompts at once | 222.7 | 220.7 | **-0.9% ±0.7%** | -0.1% |
| prefill 32K, tok/s | 5,763 | 5,884 | **+2.1%** ±1.0% | +0.3% |
| prefill 128K | 5,635 | 5,715 | **+1.4%** ±1.0% | +0.3% |
| prefill 256K | 5,401 | 5,513 | **+2.1%** ±0.4% | +0.1% |
| prefill 512K | 4,977 | 5,070 | **+1.9%** ±0.2% | +0.2% |
| prefill 1M | 4,319 | 4,398 | **+1.8%** ±0.2% | +0.3% |

r6g is at or above r6d everywhere except sixteen different prompts at once, -0.9%
(r6f: -1.8%). Against r6f it is the same bits and faster. The bench's one-prompt points
follow the text as before (code c1 +20%, prose c1 -7%).

The largest precision cost left at sixteen streams is the mHC stream projection
(`project_streams`, grid (40, 6) at 96 rows, twice a layer): 30.7 µs a call on r6d,
43.2 µs on r6f/r6g from the exact FP32 three-part split, about 1 ms a step.
