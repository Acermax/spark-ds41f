# r6f against r6d: the release comparison

Owner, 2026-10-11: before an announcement, benches on both TP4 and TP3, with enough
points to show the full over/under against the previous release (r6d).

The promotion bench showed the direction but not the full picture:
- three samples, one prompt per decode case;
- r6f's precision changes alter the text, so one prompt's acceptance swings either way
  (prose c1 -8%, code c1 +18%);
- no 1M prefill: the bench runs sizes below max-model-len - 64, and 1048576 is the
  limit itself.

## TP4 ([tp4.json](tp4.json))

r6d, then r6f (overlay arm `experiments/2026-10-11-attn-split/r6f-split.json`, the
image's sources), back to back in one session:
- **bench:** quality; decode on prose and code at 1/2/4/8/16 streams, 5-10 samples
  until each interval is within 2%; prefill on source text at 1K, 32K, 128K, 256K, 512K
  and 1M (1,000,000 tokens), two repeats each;
- **text-independent decode:**
  - the sixteen-prompt acceptance sweep, paired prompt by prompt;
  - distinct prompts at 1, 4, 8 and 16 concurrent streams;
- **latency:** short-prompt TTFT and mixed traffic;
- **determinism:** `token_determinism.py --wide`.

## TP3 ([tp3.json](tp3.json))

After re-cabling to the triangle, one job under the lab's `release-tp3` profile:
- the `release` profile at TP3's limits: 1-8 streams, prefill to 500,000 tokens
  (below the 524288 max-model-len), distinct prompts at 1, 4 and 8 streams;
- arms r6d (`r6d-tp3.json`, the r6d image) and r6f
  ([r6f-tp3-overlay.json](r6f-tp3-overlay.json): r6d-tp3 with r6f's sources mounted, the
  overlay the TP4 arms use, and the FP32 head);
- r6f's boot and `token_determinism.py --wide` are the simple test of the complete
  solution on TP3 (owner, 2026-10-11).

Open the session with `--production-config config/cluster.json`, the TP3 production
profile.

## TP4 results (2026-10-11, 18:17-19:16 UTC)

From `scripts/release_table.py r6f-release-tp4 r6d r6f`. Verdicts are against the
combined 95% intervals.

| metric | r6d | r6f | change | verdict |
|---|---|---|---|---|
| determinism (mixed15 differing) | 14/16 | 0/16, every scenario identical | | **better** |
| quality | 5/5 | 5/5 | | same |
| acceptance sweep tok/s (paired, 16 prompts) | | | -0.1% ±2.4% | same |
| acceptance sweep accepted (paired) | | | +1.2% ±4.8% | same |
| 1 distinct stream tok/s | 87.2 | 89.2 | +2.2% ±0.4% | faster |
| 4 distinct streams | 132.1 | 133.1 | +0.7% ±2.4% | same |
| 8 distinct streams | 175.8 | 175.4 | -0.2% ±1.1% | same |
| 16 distinct streams | 222.1 | 218.0 | -1.8% ±0.6% | **slower** |
| prefill 1K tok/s | 2389 | 2522 | +5.6% ±6.6% | same |
| prefill 32K | 5780 | 5842 | +1.1% ±1.0% | faster |
| prefill 128K | 5634 | 5763 | +2.3% ±1.4% | faster |
| prefill 256K | 5398 | 5490 | +1.7% ±0.3% | faster |
| prefill 512K | 4989 | 5064 | +1.5% ±0.4% | faster |
| prefill 1M | 4321 | 4377 | +1.3% ±0.2% | faster |
| short-prompt TTFT ms | 92.6 | 94.4 | +2.0% | same (no interval) |
| mixed short / long TTFT ms | 224.8 / 902.5 | 228.1 / 906.3 | +1.5% / +0.4% | same (no interval) |
| mixed stream gap p95 ms | 89.5 | 93.1 | +4.0% | slower (no interval, earlier runs ±5%) |

The bench's single-prompt points follow the text (r6f's precision changes alter it):
- prose c1 -8.5% (acceptance 1.26 to 1.06), c2-c8 +5.4 to +7.8%, c16 level;
- code c1 +18.8%, c2-c8 +1.6 to +6.3%, c16 -5.3% (acceptance 1.30 to 1.52).

Reading:
- r6f is fully deterministic (r6d: 14 of 16 mixed15 requests differ) and computes
  closer to DeepSeek's reference.
- Prefill is faster at every size from 32K to 1M, by 1.1-2.3%; single-stream decode is
  2.2% faster.
- One measured loss: 16 distinct streams, -1.8%. Real-row cost curves put r6f's
  per-step cost 1-5% above r6d's at large batches. The reference-precision pieces add
  it: FP32 head logits, the three-part FP32 mHC projection, the FP32 MoE partial path.

## TP3 results (2026-10-11, 19:33-20:26 UTC, triangle)

From `scripts/release_table.py r6f-release-tp3 r6d r6f`. r6f's boot and determinism on the
triangle are the simple test of the complete solution.

| metric | r6d | r6f | change | verdict |
|---|---|---|---|---|
| determinism (every scenario, mixed15 included) | identical | identical | | same |
| quality | 5/5 | 5/5 | | same |
| acceptance sweep tok/s (paired, 16 prompts) | | | -1.9% ±2.1% | same |
| acceptance sweep accepted (paired) | | | -2.5% ±4.6% | same |
| 1 distinct stream tok/s | 70.8 | 68.3 | -3.5% ±0.1% | **slower** |
| 4 distinct streams | 107.3 | 104.0 | -3.1% ±0.2% | **slower** |
| 8 distinct streams | 136.1 | 135.0 | -0.8% ±1.0% | same |
| prefill 1K tok/s | 2153 | 2281 | +5.9% ±13.5% | same |
| prefill 32K | 4377 | 4346 | -0.7% ±4.0% | same |
| prefill 128K | 4237 | 4260 | +0.5% ±0.7% | same |
| prefill 256K | 4060 | 4068 | +0.2% ±0.4% | same |
| prefill 500K | 3770 | 3774 | +0.1% ±0.2% | same |
| short-prompt TTFT ms | 114.8 | 119.3 | +3.9% | slower (no interval) |
| mixed short / long TTFT ms | 282.5 / 1049.9 | 289.8 / 1074.3 | +2.6% / +2.3% | same (no interval) |
| mixed stream gap p95 ms | 114.2 | 133.1 | +16.5% | slower (no interval) |

The bench's single-prompt points are all faster on r6f (prose +4.7% to +13.2%, code +1.2%
to +10.3%), with acceptance higher on every one. They follow the text.

Reading:
- On TP3, r6d was already identical in every scenario in this run, and r6f stays so.
- Prefill is level.
- Decode at one and four streams is about 3% slower. At eight streams and on the paired
  sweep it is level.
- The reference-precision work costs relatively more on three ranks than on four:
  FP32 head logits, the three-part FP32 mHC projection, the FP32 MoE partials. Each rank
  holds a third of the vocabulary and experts instead of a quarter.
- The decode gains of the bit-identical pieces measured on TP4 do not cover it here.
- The mixed-traffic p95 gap (+16.5%) has no interval and wants a repeat.
