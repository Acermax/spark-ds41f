# r6f: the promotion candidate as an image

r6f is r6e, the MoE fold and the mixed-step attention split:
- vLLM branch `r6f-attn-split` at `de6244984`, tree `4127ff975d84`;
- [r6e](../2026-10-11-r6e-candidate/README.md),
  [moe-fold](../2026-10-11-moe-fold/README.md),
  [attn-split](../2026-10-11-attn-split/README.md).

The arms so far mounted these sources over the r6d image. Promotion runs a baked image,
as r6d did.

- [vllm/series](vllm/series): r6d's series (72 entries) and the 30 commits from
  `r6d-candidate` to `de6244984` (0081-0110). Applied on the locked revision
  (`19f2c20ed`) it gives tree `4127ff975d84`, checked by `git am`. The r6d series alone
  gives `a00a6b07ff47`, r6d's image label.
- [source.json](source.json): r6d's manifest with this series: patchset `e31d5842931639a7`,
  patch head `58ea178e81f1`, tree `4127ff975d84`. Applied as `bin/spark build prepare`
  applies it (its committer, author dates), the same procedure reproduces r6d's head
  `706760d0b311`. `git diff --check` is clean.
- [upstreams.lock.json](upstreams.lock.json): r6d's lock with this series and manifest.
- [r6f-tp4.json](r6f-tp4.json): r6d-tp4 with the image `19f2c20ed4d6-r6f`, its vLLM tree
  label, this lock and `VLLM_DS41_HEAD_DTYPE=float32`; no overlays.
- [image.json](image.json): builds the image on dgx4, copies it over CX7 as r6d's gate
  did, then boots it and runs `token_determinism.py --wide`. Its references must equal
  the overlay arms' (`7ae87256d3ea` / `b8b35451dcc4`), with every scenario identical.

## Promotion (2026-10-11)

The owner promoted r6f for TP4 and TP3 on the release comparison against r6d
([results](../2026-10-11-r6f-release-bench/README.md)).

- Image: built on dgx4 from `9dfe2f1` (`build prepare` verified patch head
  `58ea178e81f1` and tree `4127ff975d84`), `sha256:b190a776`, copied over CX7 and
  verified on all four nodes.
- TP4 production on the ring since 2026-10-11 20:53 UTC
  (`cluster start --apply --replace`, ready in 82 s); `doctor --live` matches
  (weight fragmentation warnings only).
- Image gate: [image-determinism.jsonl](image-determinism.jsonl),
  `token_determinism.py --tokens 192 --long-lines 700 --wide` against the live
  service: every scenario identical, references `7ae87256d3ea` / `b8b35451dcc4`,
  equal to the overlay arms'.
- Root lock: [upstreams.lock.json](../../upstreams.lock.json) carries this series
  and [manifests/sources/2026-10-11-r6f-source.json](../../manifests/sources/2026-10-11-r6f-source.json).
- Profiles: `config/cluster-tp4.json` (this image and `VLLM_DS41_HEAD_DTYPE=float32`)
  and the three TP3 profiles (`config/cluster.json`, `-64k`, `-4k`: this image,
  threshold 4048, sparknet dispatch to 2 MiB, the FP32 head, no
  `VLLM_DS41_ENGRAM_OVERLAP`, no pinned r6 lock).
- Baseline: [2026-10-11-karmic-kraken-r6f](../../manifests/baselines/2026-10-11-karmic-kraken-r6f.json).
- Benchmark references: the release comparison's r6f arms,
  [TP4](../../manifests/benchmarks/2026-10-11-karmic-kraken-r6f-tp4.json) and
  [TP3](../../manifests/benchmarks/2026-10-11-karmic-kraken-r6f.json). Both arms
  mounted r6f's sources on the r6d image; the promoted image builds the same tree.
- TP3 runs the baked image at the next re-cabling to the triangle; the image is
  already on dgx1-3.

## TP3

[r6f-tp3.json](r6f-tp3.json) is r6d-tp3 (the triangle profile) with r6f's image, tree
label, lock and the FP32 head. Every r6e piece was validated on TP4 only. On the
triangle r6f needs the image (built and copied over the triangle's CX7 links), a lean
gate with determinism, and the TP3 recipe bench (8 streams at 512K). TP3 production is
r6 on its frozen lock.
