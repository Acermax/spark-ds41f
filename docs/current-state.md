# Current state

Production is the TP4 1M profile ([config/cluster-tp4.json](../config/cluster-tp4.json))
on the four-node ring, promoted 2026-10-12 as
[`2026-10-12-karmic-kraken-r6h`](../manifests/baselines/2026-10-12-karmic-kraken-r6h.json)
([promotion record](../experiments/2026-10-12-r6h/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6h` (`sha256:0aaa5d8b`) on all four ranks, Local
Inference Lab vLLM `19f2c20e` with r6f's series plus vLLM 0111-0117, B12X
`236ddff0`. It keeps r6f's arithmetic at DeepSeek's reference precision and its
batch-invariant temperature-0 outputs (the same references), and adds
bit-identical decode work:
- the shared expert folded into the MoE combine also when it runs on the aux
  stream (decode);
- greedy verification from vocabulary shards (no FP32 logits all-gather when
  every request in a step is greedy and none asks for logprobs);
- the vocabulary heads' token tiles vary fastest, so a call past 64 rows reads
  each block from DRAM once. Since 2026-10-12 the TP4 profile keeps the embedding
  and head in regular memory (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=0`): the carve-out
  is mapped uncached, so its second token tile could not reuse L2 (2.22 ms a
  two-tile call there, 1.73 ms in regular memory;
  `experiments/2026-10-12-r6h-loss`);
- the mHC fn split while loading, and the folded combine's early shared load;
- DSpark cost curves profiled for this image and head placement
  (`ring4-r6h-headout-20261012`).

The three-node profiles below (`config/cluster.json` and the 4 KiB / 64 KiB
variants) are promoted to r6h as well, on the same image, with their own cost
curves (`tp3-r6h-20261012`). The TileLang kernel family runs the model; B12X is
the alternative backend in the same image.

| Setting | Active value |
|---|---:|
| Kernel backend | TileLang (`kernel_backend: tilelang`): TileLang 0.1.15 + SM120/SM121 block-scaled MMA patches, DeepSeek TileKernels 2.0.0, sparknet 0.2.0 one-shot RoCE collectives; TileLang also runs RoPE, the compressor and its projection, the KV cache writers, the Engram gate and hash and the sparse-MLA metadata; B12X runs the fused WO projection, the drafter's context-KV projection, the Engram row lookup, the vision tower's kernels and the checkpoint loader |
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `19f2c20e` + the r6h series (`experiments/2026-10-12-r6h/vllm/series`, patch head `e39cd9da`, tree `68c9abd9`), B12X `236ddff0` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie, dense GEMM and prefill stage-fence patches, four-node relay and mesh transports, packed BF16 vocabulary projection, NCCL 2.30.7 + IB send-path fence and four-node ring patches |
| Image | `vllm-ds41f-kkref:19f2c20ed4d6-r6h` (`sha256:0aaa5d8b`), one digest on all ranks, built by `bin/spark build` |
| Decode tiles | block-32 FP8 decode rows on 16-, 32- or 64-row tiles by row count, TileLang's unspecialized pipeline, no K split; prefill rasterizes weights larger than L2 in panels (vLLM 0043) |
| Batch invariance | every kernel adds its reduction in a fixed order; collectives add every rank's partial in rank order with one FP32 rounding; prefill chunks leave room for every stream's decode and draft rows (threshold 4048); a mixed step runs sparse MLA per request, so decode rows keep the decode plan |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia-64k` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 524,288 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025) |
| B12X fixes (alternative backend) | indexer top-k ties by lowest logical position (B12X 0003); async-proxy fences before the TMA refill in the dense GEMM (0004) and in the BF16 and mHC prefill projections and contiguous attention (0005); W4A8 tiny decode off |
| Explicit KV memory | 3.5 GiB per rank |
| Reported KV capacity | 2,845,543 tokens (5.43x full 512K windows) |
| Memory-saver | signed DKMS 0.2.0 on all nodes; loaded UVM leaf-table packing enabled |
| Page-size profiles | 64 KiB selected; 4 KiB retains 2.2 GiB KV and a 262,144-token limit |
| Output head | exact 12-bit packed BF16 (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`; B12X 0012, vLLM 0028), bit-identical to the checkpoint; FP32 logits (`VLLM_DS41_HEAD_DTYPE=float32`, the reference's `head_dtype`) |
| Reference precision | the mHC stream projection in FP32 (three exact TF32 parts per weight), the shared expert's SwiGLU in FP32 with one rounding, MoE partials rounded once per rank; WO-A's FP8 activations are the known deviation left |
| Display carve-out | TP3 profiles (TP4 keeps both in regular memory, above): embedding and packed output-head weights in the firmware scanout reserve, loaded there directly (vLLM 0030) (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, the GPU's DRM card by PCI path, `/dev/dri/by-path/pci-000f:01:00.0-card`, as `/dev/dri/card0`); the DRM file closes after the import, so the text console keeps drawing |
| Sparse attention | TileLang sparse MLA with the MXFP4 indexer; BF16 attention (patch 0023) is available and off pending a fidelity test |
| Image input | vision tower loaded, up to 4 images per request, no host preprocessing cache |
| DSpark | 5 draft tokens, draft TP 3, adaptive verification (cost scale 2.0), dead verification rows below survival 0.2, block rejection; vocabulary-parallel greedy drafts over the checkpoint's BF16 drafter head (the target head's own tensor) and Markov projection; the drafter marks its padding rows for the MoE routers (vLLM 0042); cost curves profiled for this image with real rows (`tp3-r6h-20261012`); greedy verification from vocabulary shards when no request needs the logits (vLLM 0112-0114) |
| Collectives | sparknet one-shot all-reduce (up to 2 MiB) and all-gather over the triangle (`oneshot-direct`), larger all-reduces summed in rank order in FP32; the requested policy is enforced and a declined or lost backend fails startup (vLLM 0027, 0038) |
| Profiling context | each KV cache group gets its own blocks, in every dummy layout (vLLM 0039, 0041) |
| CUDA graphs | full, capture sizes 1-48 |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | read beside the forward launch (`SPARK3_ENGRAM_ASYNC=1`) |
| L2 weight prefetch | on (base default on SM121); TileLang scale words prefetched with each weight (vLLM 0044) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s); hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Reasoning | enabled by default; a request's `thinking` or `enable_thinking` is honored |

The release comparison against r6d is in the
[r6f results](../experiments/2026-10-11-r6f-release-bench/README.md), and r6's
acceptance benchmark in its
[promotion record](../experiments/2026-10-05-tilelang-r6/README.md); the TP3
profile's long-context admission and retrieval evidence (measured on r5o, same
shape) is in the
[memory-saver deployment](../experiments/2026-10-02-memory-saver-capacity/README.md).
Against r5p, the kernels change family; the weights, the TP3 shape and the
memory guards are unchanged.

The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [setup.md](setup.md).
