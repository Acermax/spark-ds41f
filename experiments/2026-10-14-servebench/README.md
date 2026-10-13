# servebench: steady-state load beside the burst measurements

Owner, 2026-10-14: adopt the better benchmarking techniques (MLPerf Inference's server
scenario, InferenceX's throughput-against-interactivity curves, GuideLLM and AIPerf's
goodput), built so the benchmark can later become its own project. Phase 1 is
[servebench](../../servebench/README.md): closed-loop and Poisson load with a warm-up and
drain outside the measurement window, 256 prompts, token-exact windows, batch-mean
intervals, and output tok/s split into engine steps/s x tokens per step from the server's
counters.

## Validation ([validate.json](validate.json))

On production r6h, in a lab window: closed loop at sixteen and eight requests in flight
(60 s warm-up, 300 s window, ten sub-windows), Poisson at 0.6 requests/s (about 70% of
sixteen streams' capacity) with goodput limits of 2 s to first token and 100 ms per
token, the sixteen-stream run again with the same seed (does its interval hold run to
run?), then the burst distinct-stream measurements at sixteen and eight for comparison.
