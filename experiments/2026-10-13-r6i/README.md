# r6i: sparknet main (TileLang collectives, big-core proxy) and the TileLang L2 prefetch

The one-shot all-reduce is 15-16 ms of a 90 ms sixteen-stream TP4 step (17%, every rank
alike: [r6h-loss](../2026-10-12-r6h-loss/README.md)), and a larger share on TP3, where
r6f's FP32 MoE partials doubled its bytes. r6h serves sparknet `f73a3ce`. sparknet main
(`b61660f`, 0.3.0) adds two things measured on the probe but never in serving
(`docs/oneshot.md` in sparknet):

- **The TileLang kernel family** (`SPARKNET_ROCE_KERNELS=tilelang`): bit-identical to
  the CuTe kernels, 3-5% faster at 480 KiB on the ring and about 1% at 2 MiB on a pair,
  level after the residency bound (`2aa3de0`).
- **Big-core proxy placement** (`SPARKNET_ROCE_PROXY_CPU=big`): the proxy thread polls a
  host doorbell, and on the GB10 the scheduler may put it on a Cortex-A725. Unpinned, a
  10 KiB all-reduce spread over 16-20 us; confined to the X925 cores, 17.8-18.4 us.

The third switch is the migration's C1: the TileLang L2 prefetch
(`VLLM_L2_PREFETCH_KERNELS=tilelang`, already in the r6h image), which passed its kernel
gate and screened level on r6c. With C1 and C2 both on, no standalone CuTe DSL kernel
remains in the serving path; B12X's kernels are what remains of that engine.

All three switches keep the output bits: the collectives' families are bit-identical,
placement moves a host thread, and a prefetch changes no arithmetic.

## Inputs

- [upstreams.lock.json](upstreams.lock.json): the root (r6h) lock with sparknet at
  `b61660f` (version 0.3.0); [source.json](source.json): r6h's source manifest with
  sparknet's tree `ff98c92b8b3f`. vLLM's sparknet interface (`SparknetOneShotAllReduce`,
  `freeze_kernel_resolution`) is unchanged since `f73a3ce`.
- [r6i-tp4.json](r6i-tp4.json): the promoted `config/cluster-tp4.json` (head in regular
  memory, its curves) on image `-r6i`; [r6i-tilelang.json](r6i-tilelang.json),
  [r6i-tlbig.json](r6i-tlbig.json) and [r6i-full.json](r6i-full.json) add the switches
  one at a time; [r6h.json](r6h.json) is production.

## Image ([image.json](image.json))

Built on dgx4 and copied over the CX7 links like r6h's. Gate: `--wide` determinism with
and without logprobs on the defaults (CuTe, unpinned); the references must stay r6f's
(`7ae87256d3ea` / `b8b35451dcc4`).

## Window 1 ([w1.json](w1.json))

Candidate first: full, r6h, sparknet main alone, + TileLang, + big-core proxy, full
again. Each arm: `--wide` determinism, six samples of sixteen and eight different
prompts, four of four. All arms share the production cost curves, so the switches are
the only variables. A release comparison against r6h follows if the full candidate is
level or better everywhere.
