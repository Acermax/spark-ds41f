# MoE: one rounding of each rank's partial (r6d)

The follow-up to [moe-fp32](../2026-10-11-moe-fp32/README.md), from its window 1.

Under TP4 the TileLang MoE rounded a rank's partial twice before the cross-rank sum:
- the combine wrote BF16;
- the runner then added the shared expert's partial in BF16.

DeepSeek's reference rounds a row-parallel partial once. Its `RowParallelLinear`
takes the linear's BF16 output, all-reduces it in FP32 and rounds once.

## Why not FP32 across ranks (moe-fp32 window 1, 2026-10-11)

- **Bits:** the fabric job's FP32 reduction was bit for bit equal to the rank-order
  sum at every size (1-300 rows all-reduced, 2048-16384 rows reduce-scattered).
- **Prefill cost:** its reduce-scatter took twice the BF16 time:
  - 1.97 against 3.41 ms at 2048 rows;
  - 9.88 against 19.61 ms at 16384 rows.

  Over 40 MoE layers that is about 14% of a 16K prefill, past the speed floor.
- **Decode timings:** the job's small all-reduce timings were host-bound (hundreds of
  microseconds) and say nothing about decode. Its harness needs graph timing before
  it is used for that again.

FP32 across ranks would also go past the reference's own convention for row-parallel
layers. The kernel bundle also caught a combine bug: without a shared input the
combine passed the FP32 output as its BF16 `Shared` operand. The arm never booted.

## The change (vLLM branch `r6d-moe-1r`, on r6d's tree)

- `33ef3db6a` (module, switch off):
  - The combine and MoE plans take an output dtype, and the unread `Shared` operand
    is any BF16 rows.
  - `TileLangExperts` reports FP32 under the switch, and the MoE method reports its
    experts' dtype.
  - The runner adds the shared expert to a routed output wider than the layer's
    dtype in FP32 and rounds the rank's partial once, before the all-reduce or into
    the reduce-scatter buffer. The reduction stays BF16, with no change in traffic.
- `dc271e146` (switch): `MOE_FP32_COMBINE = True` under the TileLang family.

## Window 1 ([w1.json](w1.json))

1. Kernel bundle: the MoE tests in the r6d image. The FP32 combine equals the BF16
   combine before its rounding and is batch-independent.
2. Lean screen bracketed by r6d, with `--wide` determinism and the acceptance sweep.

Acceptance:
- temperature-0 identical within the arm in every scenario;
- step time and prefill level with r6d (no traffic change; one rounding kernel fewer
  on the all-reduce path);
- paired acceptance level.
