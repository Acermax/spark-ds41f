"""WO-A's W8A16 GEMM (BF16 activations on block-32 FP8 weights, TP4: two groups of
1024 x 4096) across tile widths, pipeline stages and thread counts, without warp
specialization, at decode and prefill row counts, warm and cold under CUDA graphs.

Every config must give the default config's bits at every size (the K order is the
same, so a row's result must not depend on the tile), and the default config must be
repeatable under graph replays. Prints a table per size, the fastest config, and the
B12X whole-WO times of WO window 2 for context. Exploratory: prints "N passed" when
the bits agree and replays repeat.
"""
import itertools
import sys

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call, repeatable  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.gemm import fp8w_bf16_gemm  # noqa: E402

N, K, GROUPS = 1024, 4096, 2
SIZES = (1, 6, 16, 48, 96, 512, 2048, 8192)
TILE_M = {16: 16, 32: 32, 64: 64, 128: 128}
B12X_WO = {1: 30.75, 6: 30.62, 16: 31.06, 48: 38.06, 96: 59.02, 512: 213.8, 2048: 792.0, 8192: 4882.6}
CONFIGS = [dict(block_N=bn, num_stages=st, threads=th)
           for bn, st, th in itertools.product((16, 32, 64), (2, 3, 4), (64, 128))]
DEFAULT = {16: dict(block_N=32, num_stages=2, threads=128), 32: dict(block_N=32, num_stages=2, threads=128),
           64: dict(block_N=64, num_stages=2, threads=128), 128: dict(block_N=64, num_stages=3, threads=128)}

gen = torch.Generator(device=DEVICE).manual_seed(3)
w = torch.randint(0, 256, (GROUPS * N, K), generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8)
w[(w & 0x7F) == 0x7F] = 0x3C  # no NaN encodings
weight = w.view(torch.float8_e4m3fn)
exps = torch.randint(118, 130, (GROUPS * N // 32, K // 32), generator=gen, device=DEVICE,
                     dtype=torch.int32).to(torch.uint8)
problems, cache = [], {}


def kernel(tile, cfg):
    key = (tile, tuple(sorted(cfg.items())))
    if key not in cache:
        try:
            cache[key] = fp8w_bf16_gemm(N, K, GROUPS, block_M=tile, **cfg)
        except Exception as exc:  # noqa: BLE001
            cache[key] = exc
    return cache[key]


for rows in SIZES:
    tile = next((t for t in TILE_M if rows <= t), 128)
    a = (torch.randn((rows, GROUPS * K), generator=gen, device=DEVICE) * 0.5).bfloat16()
    ref = torch.empty((rows, GROUPS * N), dtype=torch.bfloat16, device=DEVICE)
    kernel(tile, DEFAULT[tile])(a, weight, exps, ref)
    torch.cuda.synchronize()
    results = []
    for cfg in CONFIGS:
        k = kernel(tile, cfg)
        if isinstance(k, Exception):
            continue
        out = torch.empty_like(ref)
        try:
            k(a, weight, exps, out)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            continue
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            problems.append(f"rows {rows} {cfg}: bits differ from the default config")
            continue
        warm, cold = per_call(lambda k=k, out=out: k(a, weight, exps, out), rows)
        results.append((warm, cold, cfg))
    results.sort(key=lambda r: r[0])
    print(f"rows {rows} (tile {tile}); B12X whole WO {B12X_WO[rows]:.2f} us warm")
    for warm, cold, cfg in results[:6]:
        mark = "  default" if cfg == DEFAULT[tile] else ""
        print(f"  {warm:8.2f} / {cold:8.2f} us  {cfg}{mark}")
    d = next((r for r in results if r[2] == DEFAULT[tile]), None)
    if d:
        print(f"  default {d[0]:8.2f} / {d[1]:8.2f} us; best warm {results[0][0]:.2f} ({results[0][0] / d[0]:.2f}x)")
    if rows in (1, 6, 16, 48):
        out = torch.empty_like(ref)
        k = kernel(tile, DEFAULT[tile])
        bad = repeatable(lambda: k(a, weight, exps, out), lambda: [out])
        print(f"  repeatable: {bad} of 200 replays differ")
        if bad:
            problems.append(f"rows {rows}: {bad} replays differ")

for p in problems:
    print("FAIL", p)
print(f"{0 if problems else 1} passed")
raise SystemExit(1 if problems else 0)
