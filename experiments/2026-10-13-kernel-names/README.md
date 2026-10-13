# Named TileLang kernels (r6j)

TileLang names a CUDA kernel after its `@T.prim_func`, and 39 of ours were called
`main`: every profile, trace and SASS dump showed `main_kernel`, and kernels had to be
told apart by grid signature (for example `experiments/2026-10-12-r6h-loss/group_costs.py`),
which slowed debugging more than once.

vLLM patch [0118](vllm/0118-TileLang-V4.1-name-every-kernel-after-what-it-comput.patch)
renames each prim_func after its factory: `packed_vocab_projection_kernel`,
`mhc_project_streams_kernel`, `mhc_finalize_kernel`, `moe_dispatch_kernel`,
`moe_combine_kernel`, `moe_grouped_fp8_fp4_gemm_kernel`, `router_partials_kernel`,
`sparse_mla_kernel`, `indexer_score_kernel`, `indexer_topk_segment_kernel`,
`mxfp8_gemm_kernel`, `bf16_gemm_splitk_kernel` and so on. Variants built by one factory
share its name and differ by grid, as before. The common-word names (`write`, `tokens`,
`pages`, `chunk`, `requests`, `compress`) gained a prefix. Only each prim_func's `def`
and `return` change (84 lines in 11 files); the generated code and the results do not.

## Inputs

- [vllm/series](vllm/series): r6h's series plus 0118. Replayed as build prepare does,
  the r6h part reproduces r6h's patch head `e39cd9da9938`; with 0118 the head is
  `273a6f87475e`, tree `d1f47df7170d`.
- [upstreams.lock.json](upstreams.lock.json), [source.json](source.json): r6i's (sparknet
  main) with that series.
- [r6j-tp4.json](r6j-tp4.json): the r6i release candidate (sparknet TileLang collectives,
  big-core proxy, TileLang prefetch, cost scale 1.0, its curves) on image `-r6j`.

## Image ([image.json](image.json))

Built on dgx4 and copied like r6i's. Gate: `--wide` determinism with and without
logprobs (references must stay r6f's), sixteen and eight different prompts, and a
fixed-row profile ([r6j-fixed.json](r6j-fixed.json), code16) whose kernel summaries must
show the names.
