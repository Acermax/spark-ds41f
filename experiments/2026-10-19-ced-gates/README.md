# CED ports and skip: full-model gates on the TP3 triangle

Two arms on the production r6x image and recipe (TP3, 1M), each a runtime overlay of
vLLM source with no test code:

- `ports-tp3.json`: the TileLang CED requests, row map and row copy (vLLM e4d57b3fb,
  experiments/2026-10-19-ced-kernels patches 0186-0209 without the sampling repair 0185).
- `skip-tp3.json`: the ports plus the nonfinal row skip (vLLM 3e5e91c31,
  experiments/2026-10-19-ced-skip).

**`gate.json`, determinism.** `token_determinism_text.py` with and without logprobs at two
long prompts: 700 lines (about 10.5K tokens, chunks ending at 4,048 and 8,096 both
skipped) and 2,200 lines (about 33K tokens, seven skipped chunks). Its mixed scenarios put
other streams' decode rows in the long prompt's skipped steps, which answers whether
shrinking one request's decoder rows changes another's bits. Gates:
- ports: every digest equals r6x's (experiments/2026-10-19-r6x-tp3-1m gate, 700 lines:
  62313cb5 / 9742ac2c with logprobs, 4d143977 / 12fd8257 text),
  and the ports boot through kernel preparation, so the declared copy layouts cover
  serving (an undeclared one stops its first CED step);
- skip: every digest equals the ports' at both lengths, every scenario identical.

Teacher-forced scoring cannot exercise the skip: a request with prompt logprobs keeps
every decoder row.

**`prefill.json`, the release-tp3 prefill step** (32K to 500K, three repeats; skip, ports,
skip again): the skip's prefill gain against the ports, with the repeat for drift. The
modeled ceiling at 128K is about 8-9% (experiments/2026-10-19-prefill-ceiling).

No promotion and no production change: the window restores the production profile.
