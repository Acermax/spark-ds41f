# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang Engram row lookup for DeepSeek-V4.1: FP8 table rows to BF16 embeddings.

Token ``t`` and head ``h`` read hash ``row = hashes[t, h]``. When ``t`` is live (below
the step's token count) and ``row`` lies in this rank's shard ``[start, end)`` of a
``rows``-row table, the 256 E4M3 values of the row times their E8M0 scales (one byte
per 32 values, ``2^(e - 127)``; byte 0 is ``2^-127``, byte 255 NaN) are rounded once
to BF16 into ``out[t, 256 h:256 (h + 1)]``; every other row of the first ``prepared``
tokens is zero. The arithmetic is B12X's Engram lookup (a Triton kernel of one program
per token and head).

``compact`` tables hold one row per (token, head) slot, ``t * 24 + h`` (the disk
reader stages the step's rows there); otherwise the weight holds the shard's rows,
indexed ``row - start``. ``resident_scales`` reads the scales by shard row even when
the weight is compact. Each thread converts sixteen values: one 16-byte load, two
16-byte stores. Small steps spread over more, smaller blocks (two (token, head) pairs
a 32-thread block up to ``SMALL_TOKENS``): every thread computes the same values
either way, so the launch shape does not change a bit.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

HEADS = 24
WIDTH = 256  # values per row
SF_BLOCK = 32  # values per E8M0 scale byte
_VEC = 16  # values per thread
_LANES = WIDTH // _VEC  # threads per (token, head)
SMALL_TOKENS = 48  # steps up to this many tokens use two pairs a block


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_lookup_kernel(
    max_tokens: int,
    rows: int,
    start: int,
    end: int,
    compact: bool,
    resident_scales: bool,
    pairs: int = 16,
):
    """``out[t, h]`` for the first ``P`` tokens (the ``hashes`` view's rows)."""
    threads = pairs * _LANES
    P = T.dynamic("P")  # tokens to write: the prepared count
    weight_rows = T.dynamic("weight_rows")
    scale_rows = T.dynamic("scale_rows")
    out_rows = T.dynamic("out_rows")

    @T.prim_func
    def engram_lookup(
        weight: T.Tensor((weight_rows, WIDTH), T.float8_e4m3fn),
        scales: T.Tensor((scale_rows, WIDTH // SF_BLOCK), T.uint8),
        hashes: T.Tensor((P, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        out: T.Tensor((out_rows, HEADS * WIDTH), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(P * HEADS, pairs), threads=threads) as bx:
            tx = T.get_thread_binding()
            pair = bx * pairs + tx // _LANES
            c0 = (tx % _LANES) * _VEC
            quant = T.alloc_local((_VEC,), T.float8_e4m3fn)
            result = T.alloc_local((_VEC,), T.bfloat16)
            if pair < P * HEADS:
                t = pair // HEADS
                head = pair % HEADS
                live_count = num_tokens[0]
                live = (t < live_count) & (live_count >= 0) & (live_count <= max_tokens)
                row = T.if_then_else(live, hashes[t, head], T.int64(-1))
                local = live & (row >= start) & (row < end) & (row < rows)
                shard_row = T.if_then_else(local, row - start, T.int64(0))
                weight_row = T.Cast(T.int64, pair) if compact else shard_row
                scale_row = shard_row if resident_scales else weight_row
                if local:
                    for v in T.vectorized(_VEC):
                        quant[v] = weight[weight_row, c0 + v]
                    exponent = T.Cast(T.uint32, scales[scale_row, c0 // SF_BLOCK])
                    scale = T.if_then_else(
                        exponent == T.uint32(0),
                        T.float32(2.0**-127),
                        T.if_then_else(
                            exponent == T.uint32(255),
                            T.reinterpret(T.uint32(0x7FC00000), T.float32),
                            T.reinterpret(T.shift_left(exponent, 23), T.float32),
                        ),
                    )
                    for v in T.unroll(_VEC):
                        result[v] = T.Cast(
                            T.bfloat16, T.Cast(T.float32, quant[v]) * scale
                        )
                else:
                    for v in T.unroll(_VEC):
                        result[v] = T.Cast(T.bfloat16, T.float32(0))
                for v in T.vectorized(_VEC):
                    out[t, head * WIDTH + c0 + v] = result[v]

    return engram_lookup


def engram_lookup(
    weight: torch.Tensor,
    scales: torch.Tensor,
    hashes: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    prepared: int,
    *,
    rows: int,
    start: int,
    end: int,
    compact: bool,
    resident_scales: bool,
    clear_tail: bool = False,
) -> torch.Tensor:
    """B12X's Engram lookup contract: ``out`` rows ``[0, prepared)`` written (zero
    where a slot is not live or not in the shard), ``[prepared, ...)`` zeroed when
    ``clear_tail``. ``hashes`` is ``[max_tokens, 24]`` int64, ``scales`` E8M0 bytes
    (``uint8`` or ``float8_e8m0fnu``)."""
    max_tokens = hashes.shape[0]
    if not 0 <= prepared <= max_tokens:
        raise ValueError(f"prepared tokens {prepared} outside [0, {max_tokens}]")
    if prepared:
        pairs = 2 if prepared <= SMALL_TOKENS else 16
        engram_lookup_kernel(
            max_tokens, rows, start, end, compact, resident_scales, pairs
        )(weight, scales.view(torch.uint8), hashes[:prepared], num_tokens, out)
    if clear_tail and prepared < out.shape[0]:
        out[prepared:].zero_()
    return out


__all__ = ["engram_lookup", "engram_lookup_kernel"]
