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
