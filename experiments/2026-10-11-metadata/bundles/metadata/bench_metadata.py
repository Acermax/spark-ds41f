"""Port T1: the sparse-MLA metadata kernels, vLLM's Triton kernels against the TileLang
ones, at TP4's capacity (8192 tokens, 16 requests).

The builder's request and token metadata launch eagerly before each forward, so their
time is host launch plus GPU work, measured to the end of the GPU work; the attention
layer's window and page metadata (one fused TileLang launch against Triton's two) run inside the decode graphs, so their time is GPU
time per call under CUDA graphs, warm and cold.

1. Bits: every output equals the Triton kernel's.
2. Time at decode steps (1-16 requests x 1-6 tokens) and prefill chunks.

Exits 1 on any difference or TileLang slower than Triton by more than the 3% budget.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DEVICE, per_call, slower, timing_line  # noqa: E402

from vllm.models.deepseek_v4_1.attention import _chunk, _pages  # noqa: E402
from vllm.models.deepseek_v4_1.sparse_mla import _requests, _tokens  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import metadata as tl  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

R, COLS, PAGE, WINDOW, SWA_WIDTH, MAIN_WIDTH = 16, 512, 64, 128, 128, 256
STEPS = {"c1 x1": [1], "c1 x6": [6], "c8 x6": [6] * 8, "c16 x6": [6] * 16,
         "prefill 2048": [2048], "prefill 8192 / 4": [2048] * 4}
gen = torch.Generator(device=DEVICE).manual_seed(23)
failures = []


def eager(fn, reps=200) -> float:
    """Median microseconds per call, host launch to the end of the GPU work."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    return statistics.median(times)


def outputs():
    return dict(starts=torch.empty(R + 1, dtype=torch.int32, device=DEVICE),
                rpos=torch.empty(R, dtype=torch.int64, device=DEVICE),
                counts=torch.empty(2, dtype=torch.int32, device=DEVICE),
                pos=torch.empty(CAPACITY, dtype=torch.int64, device=DEVICE),
                reqs=torch.empty(CAPACITY, dtype=torch.int32, device=DEVICE),
                slots=torch.empty(CAPACITY, dtype=torch.int64, device=DEVICE),
                visible=torch.empty(CAPACITY, dtype=torch.int32, device=DEVICE),
                swa=torch.empty(CAPACITY, SWA_WIDTH, dtype=torch.int32, device=DEVICE),
                lengths=torch.empty(CAPACITY, dtype=torch.int32, device=DEVICE),
                top=torch.empty(CAPACITY, dtype=torch.int32, device=DEVICE),
                pages=torch.empty(CAPACITY, MAIN_WIDTH, dtype=torch.int32, device=DEVICE))


table = torch.randint(0, 5000, (R, COLS), generator=gen, device=DEVICE, dtype=torch.int32)
input_slots = torch.randint(0, 10**6, (CAPACITY,), generator=gen, device=DEVICE)
outs = {"triton": outputs(), "tilelang": outputs()}

for name, lengths in STEPS.items():
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    nr, nt = len(lengths), starts[-1]
    qsl = torch.full((R + 1,), nt, dtype=torch.int32, device=DEVICE)
    qsl[: nr + 1] = torch.tensor(starts, dtype=torch.int32, device=DEVICE)
    seq = torch.zeros(R, dtype=torch.int32, device=DEVICE)
    seq[:nr] = torch.tensor(lengths, dtype=torch.int32, device=DEVICE) + 30000

    def build(impl, o):
        if impl == "triton":
            _requests[(triton.cdiv(R + 1, 128),)](qsl, seq, o["starts"], o["rpos"], o["counts"], nr, nt, R, 128)
            _tokens[(max(1, triton.cdiv(nt, 128)),)](qsl, seq, table, input_slots, o["pos"], o["reqs"], o["slots"],
                                                    o["visible"], nr, nt, table.stride(0), table.shape[1], CAPACITY,
                                                    PAGE, 1, 128, False)
        else:
            tl.requests(qsl, seq, o["starts"], o["rpos"], o["counts"], nr, nt)
            tl.tokens(qsl, seq, table, input_slots, o["pos"], o["reqs"], o["slots"], o["visible"], nr, nt,
                      max_requests=R, page=PAGE, ratio=1, circular=False)

    def layer(impl, o, rows=nt):
        if impl == "triton":
            _chunk[(rows,)](o["pos"], o["reqs"], table, o["swa"], o["lengths"], o["top"], o["visible"], qsl, o["rpos"],
                            0, table.stride(0), PAGE, WINDOW, SWA_WIDTH, False, triton.next_power_of_2(SWA_WIDTH))
            _pages[(rows, triton.cdiv(MAIN_WIDTH, 128))](o["reqs"], table, o["pages"], 0, table.stride(0),
                                                         table.shape[1], MAIN_WIDTH, 128)
        else:
            tl.decode_metadata(o["pos"], o["reqs"], table, o["swa"], o["lengths"], o["top"], o["visible"], qsl,
                               o["rpos"], o["reqs"], table, o["pages"], 0, rows, table.shape[1], page=PAGE,
                               window=WINDOW, draft=False)

    for impl, o in outs.items():
        build(impl, o)
        layer(impl, o)
    torch.cuda.synchronize()
    a, b = outs["triton"], outs["tilelang"]
    cover = max(1, triton.cdiv(nt, 128)) * 128
    same = all(torch.equal(a[k], b[k]) for k in ("starts", "rpos", "counts")) and all(
        torch.equal(a[k][:cover], b[k][:cover]) for k in ("pos", "reqs", "slots", "visible")) and all(
        torch.equal(a[k][:nt], b[k][:nt]) for k in ("swa", "lengths", "top", "pages"))
    if not same:
        failures.append(f"{name}: outputs differ")
    eb = (eager(lambda: build("triton", a)), eager(lambda: build("triton", a)))
    ep = (eager(lambda: build("tilelang", b)), eager(lambda: build("tilelang", b)))
    print(f"  builder eager  {name:18s} triton {eb[0]:7.1f} us  tilelang {ep[0]:7.1f} us  ratio {ep[0] / eb[0]:.3f}")
    if ep[0] > eb[0] * 1.03:
        failures.append(f"{name}: builder slower eager ({ep[0]:.1f} us against {eb[0]:.1f})")
    gb, gp = per_call(lambda: layer("triton", a), nt), per_call(lambda: layer("tilelang", b), nt)
    print(timing_line(nt, gb, gp) + f"  layer metadata, graphs  {name}")
    if slower(gb, gp):
        failures.append(f"{name}: layer metadata slower ({gp[0]:.2f}/{gp[1]:.2f} us against {gb[0]:.2f}/{gb[1]:.2f})")

print("summary:", "PASS" if not failures else "FAIL")
for failure in failures:
    print(" ", failure)
raise SystemExit(1 if failures else 0)
