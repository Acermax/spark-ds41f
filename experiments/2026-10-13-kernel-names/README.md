# Named TileLang kernels (r6j)

TileLang names a CUDA kernel after its `@T.prim_func`, and 39 of ours were called
`main`: every profile, trace and SASS dump showed `main_kernel`, and kernels had to be
told apart by grid signature (for example `experiments/2026-10-12-r6h-loss/group_costs.py`),
which slowed debugging more than once.

vLLM patch [0118](vllm/0118-TileLang-V4.1-name-every-kernel-after-its-file-and-w.patch)
names every kernel `<file>_<what it computes>`, or the file's own name for a file's single
kernel, so a profile line names the module that holds it (TileLang appends `_kernel`):

| file | kernels |
| --- | --- |
| `activation.py` | `activation_swiglu_clamp` |
| `cache_writer.py` | `cache_writer` |
| `collectives.py` | `collectives_rank_order_sum` |
| `compress.py` | `compress`, `compress_gather_partial_state` |
| `engram.py` / `engram_hash.py` | `engram_gate` / `engram_hash` |
| `gemm.py` | `gemm_mxfp8`, `gemm_mxfp8_partials`, `gemm_bf16`, `gemm_bf16_two`, `gemm_bf16_partials`, `gemm_bf16_splitk`, `gemm_splitk_reduce`, `gemm_splitk_reduce_two` |
| `indexer.py` | `indexer_score`, `indexer_mxfp4_quantize`, `indexer_topk_segment`, `indexer_topk_merge` |
| `metadata.py` | `metadata_requests`, `metadata_tokens`, `metadata_chunk`, `metadata_pages`, `metadata_decode` |
| `mhc.py` | `mhc_project_streams`, `mhc_finalize` |
| `moe.py` | `moe_dispatch`, `moe_combine`, `moe_grouped_fp8_fp4_gemm` |
| `rope.py` / `router.py` | `rope` / `router_partials` |
| `sparse_mla.py` | `sparse_mla` |
| `vocab.py` | `vocab_packed_projection` |
| `glm5next/nvidia/tilelang/l2_prefetch.py` (moved from `glm5next/nvidia/l2_prefetch_tilelang.py`, the same family-directory layout as `deepseek_v4_1/tilelang/`) | `l2_prefetch` |

Variants one factory builds (decode or prefill, a tile, a row bucket) share its name and
differ by grid, as before, so retuning a tile never renames a kernel. Only each
prim_func's `def` and `return` and the prefetch's path change; the generated code and the
results do not. sparknet's TileLang collectives and DeepSeek's TileKernels live in their
own packages and are not renamed here.


## Inputs

- [vllm/series](vllm/series): r6h's series plus 0118. Replayed as build prepare does,
  the r6h part reproduces r6h's patch head `e39cd9da9938`; with 0118 the head is
  `1c82f93146fb`, tree `36e5acb9fa6a`.
- [upstreams.lock.json](upstreams.lock.json), [source.json](source.json): r6i's (sparknet
  main) with that series.
- [r6j-tp4.json](r6j-tp4.json): the r6i release candidate (sparknet TileLang collectives,
  big-core proxy, TileLang prefetch, cost scale 1.0, its curves) on image `-r6j`.

## Image ([image.json](image.json))

Built on dgx4 and copied like r6i's. Gate: `--wide` determinism with and without
logprobs (references must stay r6f's), sixteen and eight different prompts, and a
fixed-row profile ([r6j-fixed.json](r6j-fixed.json), code16) whose kernel summaries must
show the names.
