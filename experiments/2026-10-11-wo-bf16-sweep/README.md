# WO-A W8A16: race fix and tile sweep

WO window 2 ([wo-bf16](../2026-10-11-wo-bf16/README.md)) found two problems with the
W8A16 WO-A GEMM.
- **Not repeatable:** graph replays at 1-48 rows gave other bits.
  - The kernel was warp-specialized. Producer warps decoded the tile's scales with
    ordinary shared stores.
  - Those stores reached the consumer warps only through the copy barriers, which order
    the asynchronous copies alone.
- **Slower than B12X:** the whole WO projection took about 46 against 30 us warm at
  decode sizes, and was slower up to 2048 rows. It was faster at 8192.

vLLM `145a64d10` turns warp specialization off for this kernel. Every thread runs the
copies, the decode and the MMA, with the pipeliner's barriers between them.

## Window 1 ([w1.json](w1.json), kernel only)

The WO tests, then [sweep_wo_a.py](sweep_wo_a.py):
- WO-A alone over tile width (16, 32, 64), stages (2-4) and threads (64, 128), at 1-8192
  rows, warm and cold;
- every config checked against the default config's bits;
- the default config's replays checked for repeatability;
- B12X's whole-WO times from window 2 printed for context.

The result chooses the tiles, or shows that WO-A needs split-K (a fixed shard count at
every row count, so a row's bits do not depend on the batch) to reach B12X.

### Window 1 (2026-10-11, 12:21 UTC)

The WO tests passed (9), so the kernel without warp specialization is correct, but the
sweep gave no timings.
- One config made an illegal memory access. It poisoned the process's CUDA context,
  every later launch failed, and the repeatability check raised the sticky error.
- Several 16-wide tiles cannot be partitioned over four warps; TileLang refuses them.

Window 2 ([w2.json](w2.json)) runs every config in its own process, so a bad config is
named and loses only itself.
