# FP32 output-head logits (r6d)

Owner, 2026-10-11: compute at native precision, never lower; speed never below
B12X. The r6d audit against DeepSeek's reference (`inference/model.py`: the head
computes FP32 logits) found production rounding every logit to BF16: the recipe sets
`VLLM_DS41_HEAD_DTYPE=model` because under upstream's FP32 default the logits
processor skipped our exact packed and TileLang heads and widened BF16 logits from
another path (r6c's rebase notes). BF16 logits step by up to 1/8; at temperature 0
two near-tied logits can round to one value, and the lowest token id wins, so a
token can differ from the reference's choice. The same setting covers the DSpark
drafter: its logits and Markov bias were BF16 too (the reference adds them in FP32).

## The change (vLLM branch `r6d-fp32-head`, on r6d's tree)

- `93eb6686c` (module): the packed and BF16 TileLang vocabulary heads take an output
  dtype. They already accumulate in FP32; with an FP32 head dtype they write FP32
  logits, and the logits processor uses them instead of widening. Heads compile for
  the model config's head dtype, so `model` keeps today's BF16 kernels.
- Switch: `VLLM_DS41_HEAD_DTYPE=float32` in the arm's recipe (upstream's default).

Expected cost: twice the logits bytes (written once per step and all-gathered across
the vocabulary shards).

## Window 1 ([w1.json](w1.json))

The vocabulary tests in the r6d image (FP32 logits rounded to BF16 equal the BF16
head's bits; closer to FP64 than BF16; batch-invariant; the processor returns FP32),
then the lean screen bracketed by r6d with the temperature-0 checks and reference
digests. Outputs may legitimately differ from r6d's where BF16 rounding had flipped
a choice; each arm must still be identical to itself in every scenario.

Acceptance: temperature-0 identical within the arm; decode and prefill at or above
r6d (and so above B12X); DSpark acceptance per step reported.
