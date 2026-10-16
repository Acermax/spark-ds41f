# r6o: r6n with step-aware Engram rows

## Hypothesis

The first Engram layer's wait for its disk rows is prefill time on both topologies.
On TP3 it is 53 / 86 / 119 ms per 16K prefill on the three ranks
([experiments/2026-10-16-tp3-prefill](../2026-10-16-tp3-prefill/README.md)), because
table 0's rows are read only after the GPU hashes the step at its start, about 50 ms
before layer 1. Reading the rows of the tokens the host already knows before the step
starts should remove most of that wait. It should change no output bit and cost decode
nothing.

## Intended delta

r6o is r6n ([experiments/2026-10-15-r6n](../2026-10-15-r6n/README.md)) with four vLLM
changes on top of its series:

1. **0148, step-aware Engram rows.** The runner prepares a step a step ahead of the GPU,
   and a prefill chunk's token ids are on the host by then. The model state keeps each
   request's prefill tokens and hashes the chunk's rows on the host
   (`engram_rows/host_hash.py`). It uses the TileLang hash's arithmetic: wrapping
   multiplies, XOR, floor remainder. The step carries those IDs as its early IDs. Rows
   of tokens the GPU samples are marked unknown and still come from the GPU's hash.
   One reader serves both. It reads the early rows as soon as the GPU has gathered the
   previous step from the same buffers; the gather now ends with a one-thread release
   of a CONSUMED sequence word. It then waits for the GPU's IDs only if a lookup is
   unknown, reads those rows, and publishes the step once. So:
   - a step of prompt rows is ready before the GPU reaches it;
   - a decode step behaves exactly as before;
   - a mixed step reads its prompt rows early and its decode rows late.

   Adaptive verification gives verifying requests GPU-chosen row counts. They stay at
   the front of the batch with the host's total, so every later request starts at the
   same row on both sides. The host refuses a step where a verifying request follows a
   prefill chunk. `SPARK3_ENGRAM_CHECK_EARLY_IDS=1` (debug, off in serving) compares
   every early ID with the GPU's and names the first difference.
2. **0147, pinned checkpoint copies.** This fixes the Engram model test that spun in a
   host-to-device copy during r6n's qualification (below).
3. **0146, the Engram model test.** It compares the readers' counters once they are
   idle. This change was in r6n's test bundle but not in its series.
4. **0143's credit.** huxiaofengtiger's forum report of r6h's TP3 startup failures is
   credited with a `Reported-by:` trailer.

Source: vLLM tree `799c7fac` (patch head `38cc5972`, as the build replays it). sparknet,
B12X, NCCL, TileLang and TileKernels are as in r6n.

## The copy that never finished

r6n's qualification left one failure unexplained: the first Engram model test spun for
37 minutes in a host-to-device copy while it loaded its checkpoint. A later diagnostic
passed. On 2026-10-16 the failure reproduced every time on dgx4, with r6n's own files,
and outside pytest.

With driver R580 (580.178.04) on the 64 KiB-page kernel (7.0.0-1019-nvidia-64k), a
pageable host-to-device copy never completes when all three of these hold:

- the copy is larger than about 128 KiB (256 KiB already hangs);
- its source is file-backed pages the process has not touched yet;
- it does not start on a 64 KiB page boundary.

The host spins in `cuMemcpyHtoDAsync_v2` inside libcuda. The GPU is idle, and nothing
is logged: no Xid, no SMMU event.

GB10 reads pageable memory through the host page tables. A 40-line script reproduces
the hang without vLLM. These cases all complete: anonymous memory at any offset, pages
the process has already read, a page-aligned start, and copies of 128 KiB or less.
Dirty or written-back pages make no difference, and neither does a shared or private
mapping.

safetensors' `get_tensor` returns a view into the mmapped file at the tensor's offset.
The test's 3.9 MB weight met every condition. `copy_weight`'s plain copy now pins a
pageable CPU source first: the CPU reads the file, and the transfer is a DMA from
pinned memory. Every safetensors-style reader goes through `copy_weight`. Production's
`--load-format b12x` installs a weight writer and never reaches this code. A new test
copies from an untouched mapping at offset 1504 in a child process, so a regression
fails the test instead of hanging the suite.

## Quality and safety gates

- **Unit tests (dgx4, r6n's image with the changed files mounted, 2026-10-16):**
  - weight transfer: 6 of 6, including the untouched-mapping copy;
  - the Engram model tests: 25 of 25, including all six descriptor-loading variants
    that used to hang, and the new end-to-end test, in which host-hashed rows equal
    B12X's GPU hash with the check on (a first chunk, a later chunk with history, a
    mixed decode-and-prefill step with an image token in its history, and a
    decode-only step);
  - the Engram reader and hash tests: 14 of 14. These include the host hash bit for bit
    against the TileLang kernel on full-range int64 geometry; early rows written only
    after the previous step's gather; and a host-hashed step published while the GPU is
    still busy before its staging.
- **Gate ([gate.json](gate.json)):**
  - one boot with `SPARK3_ENGRAM_CHECK_EARLY_IDS=1`, running the determinism texts on
    real prompts; any host/GPU difference fails a step;
  - then the plain build: temperature-0 determinism, greedy, and 8 and 4 distinct
    streams;
  - outputs must be byte-identical to r6n's, since the rows are the same bytes, read
    earlier.
- **Never slower than r6n** in the release comparison.

## Runs

1. **Image (dgx4).** dgx4 is outside the triangle's node map, so the image is built on
   it from main with `bin/spark --cluster-config experiments/2026-10-16-r6o/r6o-tp3.json
   build prepare` and `build image --apply`, then copied to dgx1-3.
2. [gate.json](gate.json): the check boot, then the plain build.
3. [profile.json](profile.json): the torch-profiled 16K prefill on r6o, r6n and r6o
   again. Table 0's wait kernel should fall from 53-119 ms per prefill to near zero.
4. [release.json](release.json): the release-tp3 suite on r6o, r6n and r6o again. The
   owner decides promotion.
5. TP4 at the next re-cable: the same gate and the release suite against r6n.

## Credits

huxiaofengtiger reported r6h's TP3 startup failures on the NVIDIA Developer Forums
(r6n's README has the details); vLLM 0143 now carries the credit.
