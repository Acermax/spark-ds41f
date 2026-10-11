"""Port B10: the Engram row lookup (FP8 table rows to BF16), B12X's Triton kernel
against the TileLang one, both launched as the lookup op launches them, for one Engram
layer at TP4's capacity (8192 tokens, 24 heads).

Layouts: the disk table as production stages it (compact rows and scales in mapped
host memory, read by the GPU across the coherent link) and a device-resident table.

1. Bits: every output row equals B12X's (NaN only has to be NaN), at every size.
2. Time per call under CUDA graphs, warm and cold, at decode steps (1-96 tokens) and
   prefill chunks.
3. Repeatable: graph replays give the eager call's bits.

Exits 1 if any bit differs, a replay differs, or TileLang is slower than B12X by more
than the 3% budget anywhere.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DECODE, DEVICE, PREFILL, REPEAT, per_call, repeatable, slower, timing_line  # noqa: E402

from b12x.sequence._shared.disk_table import MappedHostAllocation  # noqa: E402
from b12x.sequence.engram._kernels import _lookup  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.engram_lookup import HEADS, WIDTH, engram_lookup  # noqa: E402

ROWS, START, END = 4_000_000, 1_000_000, 2_000_000  # a TP4 shard of a 4M-row table
failures = []
gen = torch.Generator(device=DEVICE).manual_seed(5)
hashes = torch.randint(START - 1000, END + 1000, (CAPACITY, HEADS), generator=gen, device=DEVICE)
hashes[::17, ::5] = -1
num_tokens = torch.zeros(1, dtype=torch.int32, device=DEVICE)


def fill(t, high):
    t.copy_(torch.randint(0, high, t.shape, generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8))
    return t


owners = []
layouts = {}
for name, mapped in (("disk (mapped host)", True), ("resident (device)", False)):
    weight_rows = CAPACITY * HEADS if mapped else END - START
    if mapped:
        w_alloc = MappedHostAllocation((weight_rows, WIDTH), torch.uint8, DEVICE)
        s_alloc = MappedHostAllocation((weight_rows, WIDTH // 32), torch.uint8, DEVICE)
        owners += [w_alloc, s_alloc]
        weight, scales = w_alloc.device_view, s_alloc.device_view
    else:
        weight = torch.empty((weight_rows, WIDTH), dtype=torch.uint8, device=DEVICE)
        scales = torch.empty((weight_rows, WIDTH // 32), dtype=torch.uint8, device=DEVICE)
    fill(weight, 256)
    fill(scales, 140)[::9, 0] = 0
    layouts[name] = (weight.view(torch.float8_e4m3fn), scales, mapped)

for name, (weight, scales, compact) in layouts.items():
    print(name)
    base_out = torch.zeros((CAPACITY, HEADS * WIDTH), dtype=torch.bfloat16, device=DEVICE)
    port_out = torch.zeros_like(base_out)
    for tokens in (*DECODE, *PREFILL):
        num_tokens.fill_(tokens)

        def base(tokens=tokens):
            _lookup[(tokens, HEADS, 1)](weight, scales, hashes, num_tokens, base_out, tokens,
                                        T=CAPACITY, ROWS=ROWS, START=START, END=END,
                                        COMPACT=compact, RESIDENT_SCALES=False)

        def port(tokens=tokens):
            engram_lookup(weight, scales, hashes, num_tokens, port_out, tokens, rows=ROWS,
                          start=START, end=END, compact=compact, resident_scales=False)

        base()
        port()
        torch.cuda.synchronize()
        nan = base_out[:tokens].isnan()
        if not (torch.equal(port_out[:tokens].isnan(), nan) and torch.equal(
                port_out[:tokens].view(torch.int16)[~nan], base_out[:tokens].view(torch.int16)[~nan])):
            failures.append(f"{name} {tokens} tokens: bits differ")
        b, p = per_call(base, tokens), per_call(port, tokens)
        print(timing_line(tokens, b, p))
        if slower(b, p):
            failures.append(f"{name} {tokens} tokens: slower ({p[0]:.2f}/{p[1]:.2f} us against {b[0]:.2f}/{b[1]:.2f})")
        if tokens in REPEAT and (bad := repeatable(port, lambda: [port_out])):
            failures.append(f"{name} {tokens} tokens: {bad} replays differ")

print("summary:", "PASS" if not failures else "FAIL")
for failure in failures:
    print(" ", failure)
raise SystemExit(1 if failures else 0)
