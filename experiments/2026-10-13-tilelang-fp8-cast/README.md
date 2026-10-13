# The reference's activation cast inside the TileLang FP8 GEMM, and WO-B's decode tiles

Every TileLang block-32 FP8 projection runs TileKernels' `per_token_cast` (BF16 to E4M3
with 32-element UE8M0 scales) as its own launch, then the GEMM reads the FP8 copy back.
B12X quantizes inside its GEMM. At prefill sizes the separate pass is DRAM-bound (231 us
of WO at 2048 rows, [2026-10-09-tilelang-migration](../2026-10-09-tilelang-migration/README.md)),
which is why the TileLang WO-B and DSpark context-KV (B6) ports lose to B12X at 512-2048
rows. The cast is the reference's own quantization (byte-identical to the checkpoint's
`act_quant`: [check_cast_reference.py](../2026-10-13-tilelang-wo/check_cast_reference.py)),
so moving it into the GEMM is only legitimate if the bytes stay the same.

## The kernel (vLLM branch `mxfp8-fused-cast`, `1e0f41852`, on `wo-bf16-r6j`)

`gemm_mxfp8_cast` takes BF16 activations. Each K block is copied to shared memory and
quantized there as `act_quant` does: per 32 elements, amax floored at 1e-4, the scale
`2^ceil(log2(amax / 448))` from the exponent bits, `clamp(x / scale, -448, 448)` as
E4M3, the four exponents of 128 elements packed into one UE8M0 word. Then
`gemm_mxfp8`'s block-scaled MMA, the same K order and the same shard sum.

## WO-B's decode tiles

WO-B (5120 x 2048 per TP4 rank) has no tuned decode tiles and no wide decode tile, so
past 64 rows it takes the 128 x 128 prefill tile: in the WO window
([2026-10-13-tilelang-wo](../2026-10-13-tilelang-wo/README.md)) WO-B went from 17.8 to
32.7 us between 64 and 72 rows.

## Window 1 ([w1.json](w1.json), kernel only, dgx2)

[bundles/tuning](bundles/tuning):
1. [check_gemm_cast.py](check_gemm_cast.py) `--time`: the fused kernel against
   `per_token_cast` + `gemm_mxfp8`, bit for bit, at the B6 (512 x 5120, shards 1 and 5)
   and WO-B shapes, 1-8192 rows; time per call of both.
2. [sweep_fp8_decode.py](sweep_fp8_decode.py): WO-B's decode tile per height (16, 32,
   64, 96, 128) against today's path, warm and cold, bits against the prefill GEMM.

### Window 1 results (2026-10-13, 16:45-16:48 UTC)

- **The fused kernel did not compile.** The cast loop reused the name of the scale-staging
  loop's parallel index (`w`), which TileLang treats as immutable outside that loop; and
  the hand-written pipeline stage list did not match the statements TileLang's planner
  counts. Fixed in vLLM `e088944d1` and `3ef4c4752` (the planner stages the loop from
  `num_stages`). Checked on dgx4 the same afternoon: **bit-identical** to
  `per_token_cast` + `gemm_mxfp8` at the context-KV shape (one and five shards) and WO-B,
  1-8192 rows.
- **WO-B's decode tiles** (every candidate bit-identical to the prefill GEMM):

  | rows | best tile | against today |
  | --- | --- | ---: |
  | 1-32 | today's | 1.00 |
  | 48-64 | 64 rows, `block_N` 128, `block_K` 128, 2 stages | **0.87** |
  | 72-96 | a 96-row tile, `block_N` 64, `block_K` 128, 2 stages | **0.91** |
  | 112-128 | (128-row tile) | 1.08: keep the prefill tile |

## Window 2 ([w2.json](w2.json), kernel only, dgx2)

[sweep_gemm_cast.py](sweep_gemm_cast.py): the fused kernel's tiles at 256-8192 rows against
production's cast + prefill GEMM, warm and cold, bits checked, at both shapes.
