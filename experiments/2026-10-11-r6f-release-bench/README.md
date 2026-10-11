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

## TP3

After re-cabling to the triangle: the same plan with `r6d-tp3.json` and
`r6f-tp3.json` at TP3's limits (8 streams, 512K), and one gate of the complete r6f
(boot and determinism) first.
