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

### Window 1 results (2026-10-11, TP4, r6d / FP32 head / r6d again)

Kernel bundle: 29 passed. Temperature 0: the FP32-head arm is identical to itself in
every scenario (c8, staggered, mixed, cached, again: 0 token and 0 logprob
differences). Its reference digests differ from r6d's (`1d242845b445` /
`9cdae64e5881` against `6659a69ad547` / `15c551803076`): FP32 logits resolve
near-ties that BF16 rounding had collapsed, as expected.

| metric | r6d | r6d again | FP32 head |
|---|---|---|---|
| prose-c1 step | 31.50 ms | 31.55 ms | 31.50 ms |
| json-nothink-c1 step | 36.84 ms | 36.96 ms | 36.86 ms |
| json-nothink-c1 tok/s | 102.11 | 101.82 | 102.08 |
| prefill 1K / 16K tok/s | 2620 / 5627 | 2635 / 5605 | 2639 / 5633 |
| c8 distinct prompts tok/s | 175.86 ±0.8% | 175.56 ±1.2% | 175.22 ±0.1% |
| mixed traffic short TTFT / gap p95 | 222.3 / 86.1 ms | 223.4 / 89.4 ms | 225.0 / 85.9 ms |

Step time is level; the extra logits bytes are about 5 us a step (~0.02%). Throughput
that depends on the generated text moved with the text: prose-c1 64.47 tok/s (-6.1%;
accepted/draft 1.116 against 1.257) and c1 distinct 86.42 (-1.1%) are each one
prompt whose continuation changed, while json-nothink-c1 (acceptance 3.000 in every
arm) is level and the eight-prompt c8 set is within its interval. One prompt cannot
separate an acceptance shift from a different text; a many-prompt acceptance sweep
is queued with the next precision arms before this switch joins a candidate.
