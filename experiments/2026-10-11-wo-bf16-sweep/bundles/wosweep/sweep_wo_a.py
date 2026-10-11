"""WO-A's W8A16 GEMM (BF16 activations on block-32 FP8 weights, TP4: two groups of
1024 x 4096) across tile widths, pipeline stages and thread counts, without warp
specialization, at decode and prefill row counts, warm and cold under CUDA graphs.

Every config must give the default config's bits at every size (the K order is the
same, so a row's result must not depend on the tile), and the default config must be
repeatable under graph replays. Each config runs in its own process (one bad config cannot poison
the CUDA context of the others: window 1 lost every timing to one illegal access);
the parent prints a table per size, the fastest config, and B12X's whole-WO times of
WO window 2 for context. Exploratory: prints "N passed" when the bits agree and
replays repeat.
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

import json  # noqa: E402
import subprocess  # noqa: E402


def child(cfg_json: str) -> None:
    """One config at every size: JSON lines of times and an output digest per size."""
    cfg = json.loads(cfg_json)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    w = torch.randint(0, 256, (GROUPS * N, K), generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x3C  # no NaN encodings
    weight = w.view(torch.float8_e4m3fn)
    exps = torch.randint(118, 130, (GROUPS * N // 32, K // 32), generator=gen, device=DEVICE,
                         dtype=torch.int32).to(torch.uint8)
    for rows in SIZES:
        tile = next((t for t in TILE_M if rows <= t), 128)
        a = (torch.randn((rows, GROUPS * K), generator=gen, device=DEVICE) * 0.5).bfloat16()
        conf = DEFAULT[tile] if cfg == "default" else cfg
        try:
            k = fp8w_bf16_gemm(N, K, GROUPS, block_M=tile, **conf)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"rows": rows, "error": str(exc).splitlines()[0][:160]}), flush=True)
            continue
        out = torch.empty((rows, GROUPS * N), dtype=torch.bfloat16, device=DEVICE)
        k(a, weight, exps, out)
        torch.cuda.synchronize()
        digest = int(out.view(torch.int16).to(torch.int64).sum().item())
        warm, cold = per_call(lambda: k(a, weight, exps, out), rows)
        line = {"rows": rows, "tile": tile, "cfg": conf, "warm": warm, "cold": cold, "digest": digest}
        if cfg == "default" and rows in (1, 6, 16, 48):
            line["replays_differ"] = repeatable(lambda: k(a, weight, exps, out), lambda: [out])
        print(json.dumps(line), flush=True)


if len(sys.argv) > 1:
    child(sys.argv[1])
    raise SystemExit(0)

runs = ["default", *CONFIGS]
records, problems = {}, []
for cfg in runs:
    proc = subprocess.run([sys.executable, __file__, json.dumps(cfg)], capture_output=True, text=True, timeout=900)
    lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0:
        err = [l for l in (proc.stderr or "").splitlines() if "Error" in l or "error" in l][-1:]
        print(f"config {cfg}: exit {proc.returncode} after {len(lines)} sizes {err}")
    for line in lines:
        if "error" not in line:
            records.setdefault(line["rows"], []).append(line)

for rows in SIZES:
    rs = records.get(rows, [])
    default = next((r for r in rs if r["cfg"] == DEFAULT[r["tile"]] and "replays_differ" in r or
                    r["cfg"] == DEFAULT[r["tile"]]), None)
    print(f"rows {rows}; B12X whole WO {B12X_WO[rows]:.2f} us warm")
    if not rs:
        problems.append(f"rows {rows}: no config ran")
        continue
    for r in rs:
        if default and r["digest"] != default["digest"]:
            problems.append(f"rows {rows} {r['cfg']}: bits differ from the default config")
    for r in sorted(rs, key=lambda r: r["warm"])[:6]:
        print(f"  {r['warm']:8.2f} / {r['cold']:8.2f} us  {r['cfg']}")
    if default:
        best = min(rs, key=lambda r: r["warm"])
        print(f"  default {default['warm']:8.2f} / {default['cold']:8.2f} us; best {best['warm']:.2f} us")
    for r in rs:
        if r.get("replays_differ"):
            problems.append(f"rows {rows}: {r['replays_differ']} replays differ")
        elif "replays_differ" in r:
            print("  repeatable: 0 of 200 replays differ")

for p in problems:
    print("FAIL", p)
print(f"{0 if problems else 1} passed")
raise SystemExit(1 if problems else 0)
