# r6m: collectives that stop when they fail

r6l plus three pieces, built to remove the failure class behind the first-request
crash after boot (experiments/2026-10-14-poisson-crash) and the startup failures of
a three-node r6h deployment reported on the NVIDIA forum (2026-10-14):

1. **No fallback in the RoCE collective policy** (vLLM 0142). In-place all-reduce,
   gather, broadcast and batched send/receive no longer reach other backends.
2. **Fail-stop that stops** (sparknet `timeout-stop`, 22c6caa, on `tilelang-only`):
   - waits time out by the GPU clock (`SPARKNET_ROCE_TIMEOUT_MS`, 10 s), not by a
     poll count of unmeasured duration;
   - a timed-out wait records the peer, its HCA lane, the time waited and the
     sequence, then traps, instead of letting the step compute on unwritten
     buffers until a NaN check fires;
   - the ring relay uses the same timeout (it had a fixed 5 s);
   - TileLang kernels only (the CuTe pair is retired).
3. **Startup that waits on the host** (vLLM 0143): ranks meet on the CPU group
   before memory profiling and after the local JIT warm-up (a monitored barrier,
   `VLLM_STARTUP_RENDEZVOUS_TIMEOUT_MS`, that names a missing rank), and a stopped
   collective's report becomes the step's error. Collectives keep one timeout from
   boot onward: a kernel compiling inside a collective phase is a defect to find
   and precompile, not a wait to lengthen (the cold-cache boot below inventories
   them).

Source: vLLM tree `5e985c34` (patch head `0285d3ff`, as the build replays it;
the replay reproduces r6l's `37e56d34`/`bc089a5c` two commits earlier), sparknet
tree `8d69b6f4`. The arms are overlays of the recipe's `tp4-ring` shape
([r6m-release.json](r6m-release.json)); against r6l's arm they change only the
image, the trees, the lock and `SPARKNET_ROCE_KERNELS`/`SPIN_LIMIT` ->
`SPARKNET_ROCE_TIMEOUT_MS` (tests/test_recipe.py).

## Runs

1. [image.json](image.json): build on dgx4, copy, gate (temperature-0 determinism,
   16 and 8 distinct streams).
2. [qualify.json](qualify.json): the collective probe through vLLM's communicator;
   sparknet's fail-stop scenarios (eager, and a graph captured under a 60 s timeout
   replayed under 2 s) on every rank, each judging its own stop time and report;
   six traced boots that each open with a lone short request (the crash's
   trigger), collecting every node's log and sparknet trace; then one boot on an
   empty TileLang cache (every rank cold), which reproduces the forum report's
   startup skew and lists what compiles inside a collective phase.
3. [release.json](release.json): the release suite against r6k.

TP3 follows once these pass: see experiments/2026-10-14-one-recipe ("Releasing TP3").
