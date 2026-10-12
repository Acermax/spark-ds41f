# mHC stream projection: the FP32 split in one pass

r6f's exact FP32 mHC projection (vLLM 0092) made `project_streams` 41% slower a call
at 96 rows (30.7 to 43.2 µs, grid (40, 6), twice a layer: about 1 ms a step), the
largest precision cost left at sixteen streams
([r6g release](../2026-10-12-r6g-release/README.md)). Per stream the kernel copied fn's
tile to shared memory, synchronized, split it in place into three TF32-exact tiles,
synchronized and ran three TF32 GEMMs.

The fused variant loads each fn value from global memory into registers, splits it
there and writes the three tiles in one pass: one shared-memory round trip and one
barrier fewer per stream, the same values and the same GEMM order, so every output
keeps its bits.

## Window 1 ([w1.json](w1.json)): kernel bundle

[bundles/project](bundles/project): both variants of `project_streams`
(`post_pre` and `pre`) at 1-96 rows, fn cold (40 distinct sets in one graph) and warm,
every output compared bit for bit.

### Window 1 results (2026-10-12, 02:47 UTC)

The fused variant kept every output's bits (the 262 reported mismatches were `R_out` in
`pre` mode, which writes no `R_out`: two uninitialized buffers; the bench now skips it).
`post_pre` per call, µs, base / fused: cold fn 29.9 / 27.9 at 32 rows, 47.2 / 43.3 at
64, 62.2 / 59.2 at 96; warm 13.1 / 11.4, 24.7 / 21.2, 32.5 / 27.7. With fn cold the call
is near memory-bound and the split's latency is not overlapped with the loads.

## Window 2 ([w2.json](w2.json))

Adds the register-prefetch variant (`mhc_pipe.py`): stream 0's fn tile is requested into
registers before the token's inputs load, each later stream's before the previous
stream's GEMMs, and each tile is split from registers into the three shared tiles: the
fn loads overlap the GEMMs, with the same values and GEMM order.
