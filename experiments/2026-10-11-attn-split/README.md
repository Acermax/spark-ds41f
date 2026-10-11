# Mixed steps: decode requests keep the decode attention plan

The [r6e bisect](../2026-10-11-r6e-bisect/README.md) traced the last temperature-0
difference to the attention:
- B12X's sparse MLA ran one plan per step, so a step holding any prefill put its
  decode requests' rows on the extend plan;
- the index selection on compressed layers likewise ran one mode per step;
- the extend plan accumulates differently from the decode plan, so a decode row's
  output depended on whether a prefill shared its step.

## The change (vLLM branch `r6f-attn-split`, `de6244984` on r6e + MoE fold `7ec14fee1`)

- The attention metadata records the leading rows of decode requests
  (`num_decode_tokens`). The batch is reordered decode-first, and requests up to the
  decode threshold count, as `is_decode` counts them.
- In a mixed step those rows take the decode index mode and the decode plan; the rest
  take the prefill modes and the extend plan.
- Decode-only and prefill-only steps are unchanged.
- Mixed steps run eagerly: their per-row metadata is an ordinary allocation and each
  plan borrows the workspace in turn, so the locked workspace does not grow.

## Window 1 ([w1.json](w1.json))

Lean screen, r6e + fold / r6e + fold + split / r6e + fold bracket, with
`token_determinism.py --wide`, distinct prompts and mixed traffic.

Acceptance:
- **references equal r6e's** (`7ae87256d3ea` / `b8b35451dcc4`). A prompt alone never
  shares a step with a prefill;
- **mixed15 and every other scenario identical**, the first fully deterministic arm;
- **speed level**, mixed traffic included: a mixed step makes one more attention and
  index launch per layer.

### Window 1 (2026-10-11, 16:29-16:46 UTC), first reading

- **Determinism:** the split arm is identical in every scenario, mixed15 included. Its
  references equal r6e's bit for bit (`7ae87256d3ea` / `b8b35451dcc4`). The fold arm in
  the same run still differs on prompt 4 (mixed and mixed15). This is the first arm
  with no temperature-0 difference.
- **Speed:** level with the fold arm at c1 (prose 62.5 against 62.7 tok/s, JSON 102.4
  both) and in prefill (1K 2625 against 2638, 16K 5719 against 5721).

## Promotion bench on r6f ([w2.json](w2.json))

The promotion bench's arguments (recipe limits, r6d's reference, determinism, the
acceptance sweep) on r6f: r6e + MoE fold + this split. It runs in the session that
benches r6d and r6e + fold, so the three read like for like.
