# r6k: B12X's WO and context-KV projection retired for TileLang at the reference's precision

Owner, 2026-10-13: match DeepSeek's reference (quantize only where it does), and work until
TileLang is well ahead so the B12X versions can be retired.

## The candidate

r6j's vLLM series plus vLLM branch `wo-retire` ([vllm/series](vllm/series), 0120-0132):

- **WO (port B1)**: TileLangWOProjection, WO-A on BF16 activations as the reference's einsum
  (B12X quantized them to FP8), WO-B the reference's MXFP8 `linear`. Sequence-parallel steps
  project through the rank-order exchange.
  - Tiles from the 2026-10-13 sweeps: WO-A split 96/128-row tiles at 65-128 rows and 96-row
    tiles inside the kernel past 128; WO-B 64- and 96-row decode tiles.
  - Per call, level with or ahead of B12X's whole WO at every size (8192 rows: 2989 against
    4524 us).
- **DSpark context-KV (port B6)**: TileLang block-32 projection with the reference's
  activation cast (0.34-0.55x B12X at 1-64 rows).
- **Fused cast (0133)**: WO-A's last kernel rounds to BF16 and applies the reference's
  activation cast itself, so WO-B reads MXFP8 straight from WO-A (2048 rows level with
  B12X, 8192 at 0.67x; decode 1-32 rows faster).
- **Sparse MLA fold (0132)**: every route folds a segment as `fma(l, b, L * a)`. Before, the
  extend route's unrolled one-segment fold was contracted by the compiler to
  `fma(l, b, a)`, so a prompt prefilled in a forward past 144 rows (sequence-parallel
  mixed15) differed by an ulp from the prompt alone. The B12X WO's FP8 input absorbed that;
  the reference-precision WO exposed it
  ([2026-10-13-tilelang-wo](../2026-10-13-tilelang-wo/README.md)).

[r6k-release.json](r6k-release.json) is r6j's release configuration with the r6k image
(vLLM tree `69108474bcc7`, patch head `aa31eb85c656`). [image.json](image.json) builds and gates it;
[release-tp4.json](release-tp4.json) is the release comparison against r6h.
