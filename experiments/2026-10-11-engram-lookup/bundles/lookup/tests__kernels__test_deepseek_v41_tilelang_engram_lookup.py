# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang Engram lookup against B12X's Triton kernel, bit for bit."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("b12x")

from b12x.sequence.engram._kernels import _lookup  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.engram_lookup import (  # noqa: E402
    HEADS,
    WIDTH,
    engram_lookup,
)

DEVICE = "cuda"
MAX_TOKENS = 320
ROWS, START, END = 5000, 1200, 3700  # a 5000-row table, this rank's shard
LAYOUTS = [  # (compact, resident_scales): resident, disk, disk with resident scales
    (False, False),
    (True, False),
    (True, True),
]


def _table(compact, resident_scales, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    weight_rows = MAX_TOKENS * HEADS if compact else END - START
    scale_rows = END - START if resident_scales or not compact else weight_rows
    weight = torch.randint(
        0, 256, (weight_rows, WIDTH), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    weight[:, :4] = torch.tensor(
        [0x00, 0x80, 0x7F, 0xFF], dtype=torch.uint8
    )  # 0, -0, NaN
    scales = torch.randint(
        110, 140, (scale_rows, WIDTH // 32), generator=gen, device=DEVICE
    ).to(torch.uint8)
    scales[0::7, 0] = 0  # 2^-127
    scales[3::11, 1] = 255  # NaN
    hashes = torch.randint(
        0, ROWS + 400, (MAX_TOKENS, HEADS), generator=gen, device=DEVICE
    )
    hashes[::5, ::3] = -1
    hashes[1::9, 2] = START  # shard edges
    hashes[2::9, 4] = END - 1
    hashes[3::9, 6] = END
    return weight.view(torch.float8_e4m3fn), scales, hashes


def _b12x(weight, scales, hashes, num_tokens, out, prepared, compact, resident_scales):
    _lookup[(prepared, HEADS, 1)](
        weight, scales, hashes, num_tokens, out, prepared,
        T=MAX_TOKENS, ROWS=ROWS, START=START, END=END,
        COMPACT=compact, RESIDENT_SCALES=resident_scales,
    )  # fmt: skip
    return out


def _tilelang(weight, scales, hashes, num_tokens, out, prepared, compact, resident):
    return engram_lookup(
        weight, scales, hashes, num_tokens, out, prepared,
        rows=ROWS, start=START, end=END, compact=compact, resident_scales=resident,
    )  # fmt: skip


def _same(a, b):
    nan = b.isnan()
    assert torch.equal(a.isnan(), nan)
    assert torch.equal(a.view(torch.int16)[~nan], b.view(torch.int16)[~nan])


@pytest.mark.parametrize("compact, resident_scales", LAYOUTS)
@pytest.mark.parametrize("prepared, live", [(1, 1), (7, 5), (96, 96), (320, 300)])
def test_lookup_equals_b12x(compact, resident_scales, prepared, live):
    weight, scales, hashes = _table(compact, resident_scales, seed=prepared + live)
    num_tokens = torch.tensor([live], dtype=torch.int32, device=DEVICE)
    shape = (MAX_TOKENS, HEADS * WIDTH)
    want = torch.full(shape, 7.0, dtype=torch.bfloat16, device=DEVICE)
    got = want.clone()
    _b12x(weight, scales, hashes, num_tokens, want, prepared, compact, resident_scales)
    _tilelang(
        weight, scales, hashes, num_tokens, got, prepared, compact, resident_scales
    )
    _same(got, want)
    # Rows past the prepared count are untouched without clear_tail.
    assert (got[prepared:] == 7.0).all()
    assert (got[live:prepared] == 0).all()


def test_clear_tail_and_graph_replays():
    compact, resident_scales = True, True
    weight, scales, hashes = _table(compact, resident_scales, seed=11)
    num_tokens = torch.tensor([40], dtype=torch.int32, device=DEVICE)
    out = torch.full(
        (MAX_TOKENS, HEADS * WIDTH), 3.0, dtype=torch.bfloat16, device=DEVICE
    )
    engram_lookup(
        weight, scales, hashes, num_tokens, out, 48,
        rows=ROWS, start=START, end=END, compact=compact,
        resident_scales=resident_scales, clear_tail=True,
    )  # fmt: skip
    want = torch.full_like(out, 3.0)
    _b12x(weight, scales, hashes, num_tokens, want, 48, compact, resident_scales)
    want[48:].zero_()
    _same(out, want)
    # Equal under CUDA graph replays with the token count changed between them.
    graph_out = torch.zeros_like(out)
    _tilelang(
        weight, scales, hashes, num_tokens, graph_out, 64, compact, resident_scales
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _tilelang(
            weight, scales, hashes, num_tokens, graph_out, 64, compact, resident_scales
        )
    for live in (64, 13, 64):
        num_tokens.fill_(live)
        graph.replay()
        expect = torch.zeros_like(out)
        _b12x(weight, scales, hashes, num_tokens, expect, 64, compact, resident_scales)
        torch.cuda.synchronize()
        _same(graph_out, expect)
