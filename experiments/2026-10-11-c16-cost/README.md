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

### Window 2 results (2026-10-11, 17:30-17:34 UTC)

Real-row profiles, verify step cost (ms) at fixed token counts: r6f is 1-5% above
r6d everywhere (32: 24.04 against 24.50; 48: 29.07 against 29.70; 56: 31.37 against
32.87; 96: 43.10 against 44.61). The draft step is 2-5% above. There is no cliff.

## Torch profile of code c16 ([c16-profile](../2026-10-11-c16-profile/README.md))

r6f's extra GPU time (+710 ms over 4096 tokens) is the routed-expert grouped GEMMs:
- gate/up: 2109 against 2558 ms;
- down: 1656 against 1938 ms.

It is per call at the same launch shape. At the 48-row capture size (288 tiles) gate/up
takes 444 µs a call on r6d and 575 on r6f. Either r6f routes more real rows inside the
same padded step, or something now contends with these GEMMs.

## Window 3 ([w3.json](w3.json))

Code c16 alone (3 samples) on r6d, r6e's prefixes p1-p4 with adaptive verification on
(the bisect arms had it off), and r6f. The first prefix that loses c16 names the piece.

## Window 4 ([w4.json](w4.json))

The bench's c16 point is one prompt on sixteen streams, so a precision change that
moves the text moves the workload: acceptance at code c16 goes 1.28 (r6d, p1), 1.41
(p2), 1.64 (p3). r6d / r6f / r6d bracket on sixteen and eight distinct concurrent
prompts (four samples) separate a throughput loss from the text.

## Window 5 ([w5.json](w5.json)): step cost by piece

Window 2 put r6f's real-row verify cost 1-5% above r6d's at 32-96 tokens. Window 5 boots
each prefix of the stack with an empty cost directory and real-row profiling
(`SPARK3_DSPARK_PROFILE_TOKENS=random`): r6d, p1-allreduce (r6d's bits), p2-moe,
p3-head, p4-precision and r6f. Each boot pins its draft and verify curves. The prefix
where the cost rises names the piece to make cheaper without changing its bits.

### Window 5 results (2026-10-11, 21:20-21:32 UTC)

Verify step cost (ms, five replays) against r6d in the same window. r6d in window 2 is
the repeat: single points move up to ±2%, so read averages over sizes.

| rows | 1 | 6 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 768 | 3072 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r6d | 14.25 | 16.39 | 19.01 | 20.10 | 22.82 | 23.88 | 28.53 | 33.48 | 43.93 | 157.4 | 532.0 |
| r6d (w2) | +1.0% | -0.8% | -1.9% | -0.8% | +0.3% | +0.6% | +1.9% | -0.6% | -1.9% | +1.4% | +0.5% |
| p1-allreduce | +1.5% | +1.5% | +0.8% | -0.9% | +2.4% | +0.2% | -0.3% | +0.6% | +0.2% | +1.2% | +5.6% |
| p2-moe | -0.5% | +2.4% | +0.4% | -1.0% | +0.2% | +1.1% | +1.9% | +3.6% | +1.0% | +7.7% | +7.7% |
| p3-head | +4.1% | +2.2% | +1.9% | +2.1% | +2.4% | +2.1% | +3.8% | +3.6% | +2.6% | +7.0% | +8.3% |
| p4-precision | +1.5% | +3.3% | +0.5% | +2.2% | +2.2% | +3.5% | +6.2% | +3.6% | +1.9% | +8.2% | +8.6% |
| r6f | +0.8% | +1.2% | -1.4% | +0.6% | +3.4% | +3.4% | +7.1% | +4.0% | +2.6% | +3.4% | +4.5% |
| r6f (w2) | -0.8% | 0.0% | -0.5% | -0.8% | +4.0% | +2.6% | +4.1% | +2.5% | +1.6% | +4.7% | +5.0% |

Each piece's increment over the prefix before it, averaged over 24-96 rows (the rows
sixteen streams verify):
- p1 (SP overlap, fused sum, rank-order all-reduce; r6d's bits): +0.6%, within noise;
- p2 (MoE one rounding): +1.0%, and +6.5% on 768-3072-row prefill chunks;
- p3 (FP32 head): +1.3%; the draft step +3% (the drafter's head writes FP32 too);
- p4 (FP32 mHC projection, reference SwiGLU): +0.6%;
- r6f's remaining pieces (B3, B4, T1, fold, split): about 0 at 24-96 rows, 1-2% cheaper
  at 6-16 rows, about 4% cheaper on prefill chunks.

r6f is level with r6d at 1-16 rows and about 3.5% dearer at 24-96, which matches the
release comparison (one stream level, sixteen different prompts -1.8%). The precision
pieces (p2-p4) carry about 3% at sixteen streams.

The c16 torch profile adds one lead it cannot separate from the text: r6f's NCCL ring
all-gathers take 759 µs a call against 573 and run more often (163 against 115), while
sparknet's one-shot all-gathers run less. FP32 logits double the gathered bytes, so steps
from 17 rows on TP4 cross the 2 MiB one-shot shard limit and go to NCCL.
