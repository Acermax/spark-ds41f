# r6n on TP3

r6n is TP4 production since 2026-10-15
([experiments/2026-10-15-r6n](../2026-10-15-r6n/README.md)). The owner recabled the
dgx1–dgx2–dgx3 triangle the same evening; the site shape is `tp3-triangle`, and
production serves r6h on TP3 from its archived record
(`experiments/2026-10-14-one-recipe/tp3-r6h-production.json`) until this release.

Nothing in r6n depends on the TP size. TP3 runs the recipe's `tp3-triangle` shape: r6h's
TP3 sizing (8 streams, 512K context), NCCL profile and cost curves
(`tp3-r6h-20261012`, as r6h benched), on r6n's image and settings.

## Arms

- [r6n-tp3.json](r6n-tp3.json): the recipe's TP3 shape.
- [r6n-tp3-dev.json](r6n-tp3-dev.json), [r6h-tp3-dev.json](r6h-tp3-dev.json): each with
  the dev-mode collective RPC for the release suite's engine grid; r6h with the
  engine-grid worker files as the TP4 releases' reference arms had.
- [r6n-tp3-traced.json](r6n-tp3-traced.json): sparknet's per-op trace.
- [r6n-tp3-coldcache.json](r6n-tp3-coldcache.json): an empty TileLang cache on every
  rank, the forum report's case (an 83 s cold-JIT skew between TP3 ranks).

## Runs, queued in order (production config: r6h's TP3 record)

1. [gate.json](gate.json): temperature 0 with and without logprobs (eight prompts, the
   fifteen-stream mix with a long prompt), and 8 and 4 distinct streams.
2. [qualify.json](qualify.json): the collective probe and sparknet's fail-stop scenarios
   (eager and graph) on every rank; three traced boots that open with a lone short
   request, then Poisson load; one boot on an empty TileLang cache.
3. [release.json](release.json): the `release-tp3` suite, r6n / r6h / r6n again, through
   `bin/spark bench --suite`. The owner decides promotion.
