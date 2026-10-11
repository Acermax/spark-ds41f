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
