# The mHC stream projection at FP32 precision (r6d)

Owner, 2026-10-11: compute at native precision, never lower; speed never below
B12X. The r6d audit against DeepSeek's reference found the mHC mixes' projection
below it: the reference computes `x.float() @ hc_fn` in FP32 with FP32 `hc_fn`
weights (`inference/model.py`, `hc_mixes`), while production's TileLang
`project_streams` ran it on TF32 tensor cores, which keep 11 of each FP32 weight's 24
significant bits. It runs twice per decoder layer at every step.

## The change (vLLM branch `r6d-mhc-fp32`, on r6d's tree)

`dfc3827e3`: each `hc_fn` value loaded into shared memory is split exactly into three
TF32-representable parts (top 11 significant bits, the next 11, the remainder of at
most 2), and the projection runs three TF32 GEMMs into one FP32 accumulator. The
streams are BF16, exact in TF32, so every product is exact: the mixes have FP32
precision. The kernel test now bounds the projection error against FP64 at 3e-5 of
the largest value, about a twentieth of what TF32 gives.

## Window 1 ([w1.json](w1.json))

The mHC tests in the r6d image, then the lean screen bracketed by r6d with the
temperature-0 checks and reference digests. Outputs may differ from r6d's (the mixes
change in their low bits); the arm must be identical to itself in every scenario.

Acceptance: temperature-0 identical within the arm; decode, prefill and mixed traffic
level with r6d (the extra GEMMs are small next to the streams' traffic).

## Status

Window 1 (2026-10-11, 07:55-08:15 UTC, r6d / mhcfp32 / r6d): the mHC tests pass in the
image, including the FP32 bound against FP64.

| Measure | r6d | mhcfp32 | r6d (end) |
| --- | --- | --- | --- |
| prose c1 step | 31.48 ms | 31.85 ms (+1.2%) | 31.58 ms (+0.3%) |
| json c1 step | 36.67 ms | 36.48 ms (-0.5%) | 36.72 ms (+0.1%) |
| single stream, distinct prompts | 87.43 tok/s | +2.4% | -0.3% |
| eight distinct concurrent prompts | 175.76 tok/s | -1.8% | -0.1% |
| cold prefill 1K / 16K | 2652 / 5613 tok/s | -0.5% / -0.8% | +0.4% / -0.3% |
| mixed: short / long TTFT | 229.4 / 902.2 ms | -1.7% / -1.5% | -2.4% / +0.1% |
| temperature 0 | identical | identical | identical |
| references against r6d | - | different | same bits |

The arm is identical to itself in every scenario; its references differ from r6d's, as
the mixes now carry FP32 precision. Tokens per second move with the generated text
(prose: accepted per draft 1.525 against 1.257), so step time is the comparison: about
+1% on prose, -0.6% on JSON, -1.8% at eight distinct streams and 0.5-0.8% on prefill
against the mean of the two r6d runs. A cost under 2%, far above the B12X floor; an FP32
FMA variant of the 24-column projection (no tensor cores) is the follow-up to recover it.
