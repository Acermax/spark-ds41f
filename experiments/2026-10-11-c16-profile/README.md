# Code c16: where r6f's step time goes

[c16-cost](../2026-10-11-c16-cost/README.md) window 2 measured both releases' target
verify and drafter step costs at fixed token counts with real rows. r6f is 1-5% above
r6d at every count, with no cliff; at 48 rows verify plus draft is about 35 ms. Yet the
bench's code c16 steps take 67 ms on r6d and 81 ms on r6f. About half of a c16 step is
outside those forward costs, and that is where r6f loses about 14 ms.

## Window 1 ([w1.json](w1.json))

Both releases with the torch profiler (`--profiler-config`), and lab profile workload
`code16`: the bench's code prompt on sixteen streams, 256 tokens, one profiled round
after a warm-up ([profile_c8.py](profile_c8.py) with a `code` case). The lab collects
every rank's trace and compares kernel time and the step's wall time between the arms:
GPU kernels against host and scheduling time.
