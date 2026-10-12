"""Packed vocabulary head per call: the base grid against token-tiles-fastest.

TP4's shard (32,320 x 5,120, about 248 MB packed) never fits L2, so every call reads
its weights from DRAM; past 64 rows a call has two token tiles. Outputs must be
bit-identical. Times are per call in a CUDA graph, microseconds.
"""
import importlib.util
import sys

import torch
from b12x.gemm import packed_bf16_vocab_projection as packed_format

N, K = 32320, 5120
NAMES = sys.argv[1:] or ["vocab_base", "vocab_swap"]
ROWS = (16, 48, 64, 65, 80, 96, 128)


import vllm.models.deepseek_v4_1.tilelang  # noqa: E402  (the copies' relative imports)


def load(name):
    qualified = f"vllm.models.deepseek_v4_1.tilelang.{name}"
    spec = importlib.util.spec_from_file_location(qualified, f"/b/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def timed(call, rounds=20, per_graph=4):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(per_graph):
            call()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * per_graph)


modules = {name: load(name) for name in NAMES}
gen = torch.Generator(device="cuda").manual_seed(0)
weight = (torch.randn(N, K, generator=gen, device="cuda") * 0.02).bfloat16()
packed = packed_format.pack(weight.clone())
failures = 0
print(f"{'dtype':>8} {'rows':>5} " + " ".join(f"{n:>14}" for n in NAMES))
for out_dtype, tl_dtype in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
    for rows in ROWS:
        x = (torch.randn(rows, K, generator=gen, device="cuda") * 0.5).bfloat16()
        outs, line = {}, []
        for name, m in modules.items():
            block_M = m.row_tile(rows)
            kernel = m.packed_vocab_projection(N, K, packed.exponent_base, block_M=block_M,
                                               out_dtype=m._OUT[out_dtype], **m.tile_config(block_M, K))
            out = torch.empty(rows, N, dtype=out_dtype, device="cuda")

            def call(kernel=kernel, out=out):
                kernel(x, packed.values, packed.exception_rows, packed.exception_entries, packed.row_offsets, out)

            line.append(f"{timed(call):14.1f}")
            call(); torch.cuda.synchronize()
            outs[name] = out
        for name in NAMES[1:]:
            if not torch.equal(outs[name].view(torch.uint8), outs[NAMES[0]].view(torch.uint8)):
                failures += 1
                print(f"MISMATCH {name} {tl_dtype} rows={rows}")
        print(f"{tl_dtype:>8} {rows:5d} " + " ".join(line), flush=True)
print("bit-identical" if not failures else f"{failures} mismatches")
sys.exit(1 if failures else 0)
