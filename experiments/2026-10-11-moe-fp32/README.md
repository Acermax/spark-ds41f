# MoE partials in FP32 until the layer output (r6d + rank-order all-reduce)

Owner, 2026-10-11: compute at native precision, never lower; speed never below
B12X. The r6d audit against DeepSeek's reference (`inference/model.py` in the
checkpoint) listed the MoE output's roundings as the last open deviation.

## The reference

- `Expert.forward`: gate and up projections widened to FP32, clamped, `silu(gate) * up`
  and the route weight in FP32, rounded to BF16 for the down projection (whose FP8
  linear quantizes the activation), BF16 out.
- `MoE.forward`: the local experts' BF16 outputs accumulate into an FP32 `y`, `y` is
  all-reduced in FP32, the shared expert's BF16 output is added in FP32, and the sum
  rounds to BF16 once.
- `RowParallelLinear` (the reference's own tensor-parallel convention): each rank's
  partial comes out of the linear in BF16, is widened, all-reduced in FP32 and rounded
  once.

The reference splits experts across ranks; r6d splits each expert's inner dimension
(2304, 576 a rank at TP4), so a rank holds every expert's partial, never a whole
expert output.

## r6d at TP4 (TileLang family)

| step | dtype | against the reference |
|---|---|---|
| route weight folded into the SwiGLU before the down projection's MXFP8 cast | FP32 to E4M3 | as the reference (which rounds to BF16 first) |
| each expert's down-projection partial (`grouped_fp8_fp4_gemm`) | BF16 | the reference's tensor-parallel convention: a partial rounded, then summed in FP32 |
| top-k combine (`moe_combine_v41`) | FP32 sum, **BF16 out** | extra rounding: the reference accumulates in FP32 |
| shared expert's partial (row-parallel FP8 linear) | BF16 | the tensor-parallel convention |
| shared + routed add (MoE runner) | **BF16** | extra rounding: the reference adds in FP32 |
| all-reduce / prefill reduce-scatter | BF16 in, FP32 rank-order sum, one rounding | its inputs were rounded twice already |

Attention's WO-B output (BF16 partials, FP32 rank-order sum, one rounding) already
matches `RowParallelLinear`.

## The change (vLLM branch `r6d-moe-fp32`, on `r6d-allreduce`)

- `3499ff055` (module, switch off): `moe_combine_v41` and the MoE plans take an
  output dtype; `TileLangExperts.output_dtype` follows
  `kernels.moe_output_dtype()`, and the MoE method reports a modular kernel's
  experts' dtype, so the runner adds the shared expert's partial and reduces across
  ranks in FP32 and `DeepseekV4MoE` rounds once. The rank-order sum's TileLang kernel
  takes FP32 sources (four a thread, one 16-byte access) for reductions past the
  one-shot limit and the prefill reduce-scatter.
- `f48fb9940` (switch): `MOE_FP32_PARTIALS = True` under the TileLang family.

The arm sits on the rank-order all-reduce because FP32 halves the rows sparknet's
one-shot path takes (102 instead of 204 at 5120 columns); without it, steps of 103
rows or more would fall to NCCL's ring. Its control is that arm
(`experiments/2026-10-11-allreduce-rank-order/allreduce.json`): one variable.

Expected cost: the MoE reduction carries twice the bytes (the attention all-reduce
does not change), each MoE layer adds one rounding kernel after it, and the combine
writes FP32 rows. Memory does not grow: the large buffer, the experts' per-route
down-projection output, stays BF16.

Not changed: keeping each rank's per-expert partial in FP32 as well would double that
buffer (about 0.5 GB a rank at the largest prefill chunk) to go past the reference's
convention, not to meet it.

Candidate assembly: the fused sequence-parallel sum (`r6d-fused-sum`) adds the
prefill reduce-scatter's BF16 parts inside its kernel; with this switch its parts
are FP32, so the fold must give that kernel FP32 sources.

## Window 1 ([w1.json](w1.json))

1. Kernel bundle in the r6d image: the rank-order sum bit for bit in BF16 and FP32
   (signed zeros, subnormals, infinities, NaN, CUDA graph replays, views of one
   buffer); the FP32 combine equals the BF16 combine before its rounding, and a
   token's FP32 output does not depend on the batch.
2. Fabric ([fabric_moe_reduce.py](fabric_moe_reduce.py), every node): the
   communicator's all-reduce at decode and CED sizes (1-300 rows) and the
   reduce-scatter at prefill chunks (2048-16384 rows), BF16 against FP32 plus the
   rounding; FP32 results bit for bit against the rank-order sum of the gathered
   partials. It prints the per-step cost bound (40 MoE layers times the difference).
3. Lean screen bracketed by the all-reduce arm, with `token_determinism.py --wide`
   (mixed15), distinct prompts, mixed traffic and the sixteen-prompt acceptance sweep.

Acceptance: temperature-0 identical within the arm in every scenario; each metric's
cost reported against the all-reduce arm and every metric still above B12X's. A
cost beyond noise goes to its own follow-up (the rounding fused into the reduction,
or the combine writing straight into the one-shot buffer), not a weaker switch.

## Window 1 results (2026-10-11, 11:18-11:22 UTC)

- **Kernel bundle:** 49 passed, 3 failed. `test_moe_experts_fp32_output` caught a
  combine bug: without a shared input the combine passed its output as the BF16
  `Shared` operand, and the arm would have crashed at its first forward. The window
  was closed before the arms booted.
- **Fabric:** every FP32 result bit for bit equal to the rank-order sum.
  - The prefill reduce-scatter took twice the BF16 time (1.97 against 3.41 ms at 2048
    rows, 9.88 against 19.61 ms at 16384), about 14% of a 16K prefill over 40 MoE
    layers.
  - The small all-reduce timings were host-bound and unusable.
- **Outcome:** superseded by
  [moe-one-rounding](../2026-10-11-moe-one-rounding/README.md). It rounds each rank's
  partial once, as the reference's row-parallel layers do, and keeps the reduction
  BF16.
