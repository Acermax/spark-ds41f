"""mHC project_streams per call, two variants, with fn cold in L2 as in serving.

Serving runs about 86 sublayer entries a step, each with its own 2 MB fn, so every
call reads fn from DRAM: calls cycle SETS distinct fn and stream sets inside one CUDA
graph; "warm" reuses one set. Every output (R_out, P, S, C, Q) must have the base
variant's bits. Times are per call, in microseconds.
"""
import importlib.util
import sys

import torch

H, SETS, WORLD = 5120, 40, 1
NAMES = sys.argv[1:] or ["mhc_base", "mhc_fused"]
ROWS = (1, 6, 12, 16, 24, 32, 48, 64, 96)
MODES = ("post_pre", "pre")


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"/b/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(calls, rounds=20):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


modules = {name: load(name) for name in NAMES}
gen = torch.Generator(device="cuda").manual_seed(0)
fns = [torch.randn(24, 4 * H, device="cuda", generator=gen) * 0.02 for _ in range(SETS)]
failures = 0
print(f"{'mode':>9} {'rows':>5} " + " ".join(f"{n + ' cold':>16} {n + ' warm':>16}" for n in NAMES))
for mode in MODES:
    kernels = {name: m.project_streams(H, mode, block_M=m.BLOCK_M, world=WORLD) for name, m in modules.items()}
    splits = H // modules[NAMES[0]].BLOCK_H
    for rows in ROWS:
        inputs = [dict(
            x=torch.randn(WORLD, rows, H, device="cuda", generator=gen).bfloat16(),
            residual=torch.randn(rows, 4 * H, device="cuda", generator=gen).bfloat16(),
            post=torch.rand(rows, 4, device="cuda", generator=gen) * 2,
            comb=torch.softmax(torch.randn(rows, 4, 4, device="cuda", generator=gen), -1),
            pre=torch.softmax(torch.randn(rows, 4, device="cuda", generator=gen), -1),
            fn=fns[i]) for i in range(SETS)]
        outs, line = {}, []
        for name, kernel in kernels.items():
            buffers = [dict(
                r_out=torch.empty(rows, 4 * H, dtype=torch.bfloat16, device="cuda"),
                p=torch.empty(splits, rows, 1, 24, device="cuda"),
                s=torch.empty(splits, rows, 1, device="cuda"),
                c=torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"),
                q=torch.empty(splits, rows, device="cuda")) for _ in range(SETS)]

            def call(i, kernel=kernel, buffers=buffers):
                a, b = inputs[i], buffers[i]
                kernel(a["x"], a["residual"], a["post"], a["comb"], a["pre"], a["fn"],
                       b["r_out"], b["p"], b["s"], b["c"], b["q"])

            cold = timed([lambda i=i: call(i) for i in range(SETS)])
            warm = timed([lambda: call(0)] * SETS)
            for i in range(SETS):
                call(i)
            torch.cuda.synchronize()
            outs[name] = buffers
            line += [f"{cold:16.2f}", f"{warm:16.2f}"]
        base = outs[NAMES[0]]
        for name in NAMES[1:]:
            for i in range(SETS):
                # "pre" reads streams already updated and writes no R_out.
                for key in (("p", "s", "c", "q") if mode == "pre" else ("r_out", "p", "s", "c", "q")):
                    if not torch.equal(outs[name][i][key].view(torch.uint8), base[i][key].view(torch.uint8)):
                        failures += 1
                        print(f"MISMATCH {name} {mode} rows={rows} set={i} {key}")
                        break
        print(f"{mode:>9} {rows:5d} " + " ".join(line), flush=True)
print("bit-identical" if not failures else f"{failures} mismatches")
sys.exit(1 if failures else 0)
