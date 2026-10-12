# r6g against r6d: the release comparison on TP4

The [release protocol](../../docs/lab.md#release-comparison) (lab profile `release`), with
r6d run again after r6g: the r6g windows measured up to 2.3% between two arms of one
configuration twenty minutes apart, so r6d before and after bounds the drift.

- r6d: `experiments/2026-10-10-r6d-deterministic/r6d-tp4.json` (the previous release).
- r6g: `experiments/2026-10-12-r6g/r6g-tp4.json`, the baked `-r6g` image
  ([r6g](../2026-10-12-r6g/README.md): r6f with the aux-stream MoE fold and greedy
  verification from vocabulary shards, both bit-identical to r6f).
