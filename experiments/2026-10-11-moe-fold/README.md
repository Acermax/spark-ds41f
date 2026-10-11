# MoE: the combine adds the shared expert's output

[MoE one rounding](../2026-10-11-moe-one-rounding/README.md) made the TileLang combine
write each rank's partial in FP32. The runner then reads it back, adds the shared
expert's BF16 output in FP32 and rounds once. That cost about 1.3% of a 16K prefill:
- the combine writes twice the bytes;
- an add kernel reads them back with the shared output, then writes BF16.

## The change (vLLM branch `r6e-moe-fold`, `7ec14fee1` on `r6e-draft2`)

When the shared expert runs before the routed experts on the same stream, the runner
hands its output to the TileLang experts. That happens on steps past the shared-stream
threshold (256 rows), which includes the sequence-parallel prefill.
- The combine adds the shared output to the slot-order FP32 sum and rounds once to BF16.
- The runner skips its add.
- Decode steps keep the shared expert on its own stream, with the runner's add.

The bits are the runner's: the same FP32 add and one round-to-nearest-even. So a row's
result does not depend on which path its step takes.
`test_moe_experts_fold_shared` checks the folded combine bit for bit against both
runner paths (the all-reduce add and the reduce-scatter buffer).

## Window 1 ([w1.json](w1.json))

1. Kernel bundle: the MoE tests at `7ec14fee1`, in the r6d image.
2. Lean screen, r6e / r6e-fold / r6e bracket, with `token_determinism.py --wide`,
   distinct prompts and mixed traffic.

Acceptance:
- the references equal r6e's bit for bit (the long prompt alone prefills
  sequence-parallel, so it takes the folded path);
- determinism as r6e;
- prefill above r6e's, with nothing else below it.
