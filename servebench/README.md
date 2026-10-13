# servebench

Steady-state load measurements for OpenAI-compatible LLM servers. Standard library only;
nothing here imports the cluster tooling, so it can move to its own repository unchanged.

```sh
python3 -m servebench run http://127.0.0.1:8000 --mode closed --concurrency 16 \
    --warmup 60 --duration 300 --output c16.json
python3 -m servebench run http://127.0.0.1:8000 --mode poisson --rate 0.6 \
    --slo-ttft 2 --slo-tpot 100 --output poisson.json
python3 -m servebench compare base.json candidate.json
```

## Why

The distinct-stream measurements it complements start N requests at once and time from
the first token to the last. That window holds a ramp (all prefills together) and a
tail (fast streams finish first, so the batch drains toward one), so a "16-stream"
number is partly an 8-, 4- and 1-stream number. Sixteen prompts also leave speculative
decoding's acceptance tied to a handful of texts, and a change in the text (a precision
change) then reads as a change in speed. The methods here follow MLPerf Inference's
server scenario, GuideLLM and AIPerf.

## Method

- **Load.** `closed` keeps N requests in flight, starting the next as each ends, so the
  server sees a steady N with new prefills mixed into decode steps. `poisson` sends
  requests at a fixed rate whatever the server does (open loop), so queueing shows as
  latency. `burst` reproduces the old shape for comparison.
- **Window.** A warm-up (default 60 s) runs before the measurement window (default
  300 s); when the window closes no new requests start, and the in-flight ones drain
  outside it.
- **Tokens.** Every streamed chunk is timed with the request's cumulative token count
  (vLLM's per-chunk usage), so the window counts exactly the tokens produced inside it.
- **Prompts.** 256 distinct prompts (sixteen task styles by 32 topics), each request with
  a fresh cache salt so repeats never hit the prefix cache, and by default exactly
  `--max-tokens` output tokens each (`ignore_eos`), so output length is not a variable.
- **Intervals.** The window is cut into equal sub-windows (default 10); each is one
  sample of throughput and of every server rate (batch means), with Student t intervals.
- **Speed and text apart.** With the vLLM adapter the server's counters are read at
  every sub-window boundary: output tok/s = engine steps/s x output tokens per step.
  The step rate is the engine's speed; tokens per step moves with draft acceptance, that
  is with the text. A precision change that alters the text shows in the second, a
  kernel change in the first.
- **Latency.** Time to first token, time per output token and per-user tok/s
  (percentiles over requests started in the window); with `--slo-ttft` / `--slo-tpot`,
  SLO attainment and goodput (requests per second meeting both limits).
- **Compare.** Batch-mean metrics get a Welch interval; percentile metrics are judged
  against a tolerance (default 3%) and marked "no interval". The comparison warns when
  the two runs' workloads differ.

## Engine cost grid

```sh
python3 -m servebench engine http://127.0.0.1:8000 --output grid.json
python3 -m servebench compare base-grid.json candidate-grid.json
```

The server times its own fixed-shape dummy steps, the speculative-decoding cost profile's
steps, at the shapes asked for. The default grid:
- verification rows for 1-16 requests at 4K and 64K context;
- one and eight requests at 256K;
- one request at the server's longest context;
- 8K prefill chunks at 0, 64K and 256K.

Each shape is warmed and then timed in rounds of alternating order (default 30). Every
build does identical work with no text, acceptance or scheduling in it, so the step times
are the engine's speed and their intervals are tight. This is the measurement that judges
kernel and engine changes; tok/s judges what users get. It needs vLLM's dev-mode collective
RPC (`VLLM_SERVER_DEV_MODE=1`) and a worker exposing `engine_grid`. It refuses a busy
server and resets the prefix cache afterwards, since dummy steps write KV slots.

## Output

`--output` writes `servebench/1` JSON: tool, server, workload (with the prompt set's
SHA-256), and summary (`output_tps`, `server`, `server_batches`, `latency`,
`inflight_mean`, `requests_per_s`, `requests`). `--save-requests` adds every request's
timeline. The last stdout line is a one-line summary (`{"summary": ...}`) for tables.

## Planned

Model behaviour per build (acceptance by draft position, teacher-forced log-probabilities:
KL divergence, top-1 agreement, perplexity), concurrency and rate sweeps (throughput against per-user speed), and precision: per-token
log-probabilities on fixed text against a baseline (KL divergence, top-1 agreement,
perplexity) as a gate before speed.
