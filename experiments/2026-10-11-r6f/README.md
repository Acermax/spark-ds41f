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

Promotion itself (root lock, profile, baseline, bench reference, docs) follows the
owner's decision.
