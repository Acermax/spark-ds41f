# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A reduce-scatter with the one-shot all-reduce's arithmetic.

sparknet's one-shot all-reduce adds the ranks' BF16 partials in FP32 in rank order
(source 0, then 1, 2, 3) and rounds once, so a row has the same bits whatever the
message size, the row's position or the rank. NCCL's reduce-scatter, which prefill
sequence parallelism uses from 205 rows, adds a chunk around the ring starting
after its owner and rounds to BF16 after every hop: a row's bits change with the
step's row count. ``reduce_scatter_rank_order`` exchanges the chunks unreduced
(NCCL send/recv on the same communicator, the same bytes) and adds them as the
one-shot all-reduce does, so a row reduces to the same bits on every path. FP32
partials (the MoE's, kept unrounded until the layer output) add the same way and
round nothing.

At four ranks only the neighbours (rank +-1, the ring4 cable order sparknet and
NCCL use) exchange: each half of the opposite rank's chunk is relayed through one
neighbour, so the four links carry equal bytes. Fewer ranks exchange directly.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

_VEC = {torch.bfloat16: 8, torch.float32: 4}  # per thread: one 16-byte access
_DTYPE = {torch.bfloat16: T.bfloat16, torch.float32: T.float32}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def rank_order_sum_kernel(
    world: int, dtype: T.dtype = T.bfloat16, vec: int = 8, threads: int = 256
):
    """``out = round(((s0 + s1) + s2) + s3)`` in FP32, element by element, for BF16
    or FP32 sources (FP32 rounds nothing) in groups of ``vec`` (16 bytes); the
    first source is taken as is (not added to zero), so signed zeros survive."""
    assert 2 <= world <= 4
    m = T.dynamic("m")  # groups of vec elements

    @T.macro
    def load(src, xs, total, row, first):
        for i in T.vectorized(vec):
            xs[i] = src[row, i]
        for i in T.unroll(vec):
            if first:
                total[i] = T.cast(xs[i], T.float32)
            else:
                total[i] = total[i] + T.cast(xs[i], T.float32)

    @T.macro
    def store(out, xs, total, row):
        for i in T.unroll(vec):
            xs[i] = T.cast(total[i], dtype)
        for i in T.vectorized(vec):
            out[row, i] = xs[i]

    if world == 2:

        @T.prim_func
        def rank_order_sum(
            s0: T.Tensor((m, vec), dtype),
            s1: T.Tensor((m, vec), dtype),
            out: T.Tensor((m, vec), dtype),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((vec,), dtype)
                total = T.alloc_local((vec,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    store(out, xs, total, row)

    elif world == 3:

        @T.prim_func
        def rank_order_sum(
            s0: T.Tensor((m, vec), dtype),
            s1: T.Tensor((m, vec), dtype),
            s2: T.Tensor((m, vec), dtype),
            out: T.Tensor((m, vec), dtype),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((vec,), dtype)
                total = T.alloc_local((vec,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    load(s2, xs, total, row, False)
                    store(out, xs, total, row)

    else:

        @T.prim_func
        def rank_order_sum(
            s0: T.Tensor((m, vec), dtype),
            s1: T.Tensor((m, vec), dtype),
            s2: T.Tensor((m, vec), dtype),
            s3: T.Tensor((m, vec), dtype),
            out: T.Tensor((m, vec), dtype),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((vec,), dtype)
                total = T.alloc_local((vec,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    load(s2, xs, total, row, False)
                    load(s3, xs, total, row, False)
                    store(out, xs, total, row)

    return rank_order_sum


def rank_order_sum(
    parts: list[torch.Tensor], out: torch.Tensor | None = None
) -> torch.Tensor:
    """Equal-shape tensors added in FP32 in list order and rounded once: BF16 or
    FP32 CUDA tensors whose element count fills 16-byte groups go through the
    TileLang kernel, anything else through the same arithmetic in torch."""
    if out is None:
        out = torch.empty_like(parts[0])
    n, dtype = parts[0].numel(), parts[0].dtype
    vec = _VEC.get(dtype)
    if (
        parts[0].is_cuda
        and vec is not None
        and out.dtype == dtype
        and 2 <= len(parts) <= 4
        and n % vec == 0
        and all(p.is_contiguous() for p in parts)
        and out.is_contiguous()
    ):
        if n:
            rank_order_sum_kernel(len(parts), _DTYPE[dtype], vec)(
                *(p.view(-1, vec) for p in parts), out.view(-1, vec)
            )
        return out
    total = parts[0].to(torch.float32, copy=True)  # never add into an input
    for part in parts[1:]:
        total += part.float()
    return out.copy_(total)


def exchange_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> list[torch.Tensor]:
    """Every rank's chunk ``rank`` of ``input_tensor`` (contiguous, ``world * L``
    rows), unreduced and in rank order; this rank's own is a view of the input.

    ``comm`` is the group's PyNCCL communicator (``send``, ``recv``,
    ``group_start``, ``group_end``)."""
    chunks = input_tensor.view(world, -1, *input_tensor.shape[1:])
    parts = torch.empty_like(chunks)
    if world == 4:
        nxt, prv, opp = (rank + 1) % 4, (rank + 3) % 4, (rank + 2) % 4
        half = -(-chunks.shape[1] // 2)
        # Halves of the opposite rank's chunks, in transit: from the previous rank
        # for the next one (rows :half), from the next rank for the previous one.
        relay_a = torch.empty_like(chunks[0, :half])
        relay_b = torch.empty_like(chunks[0, half:])
        comm.group_start()
        _send(comm, chunks[nxt], nxt)
        _send(comm, chunks[opp, :half], nxt)
        _send(comm, chunks[prv], prv)
        _send(comm, chunks[opp, half:], prv)
        _recv(comm, parts[prv], prv)
        _recv(comm, relay_a, prv)
        _recv(comm, parts[nxt], nxt)
        _recv(comm, relay_b, nxt)
        comm.group_end()
        comm.group_start()
        _send(comm, relay_a, nxt)
        _send(comm, relay_b, prv)
        _recv(comm, parts[opp, :half], prv)
        _recv(comm, parts[opp, half:], nxt)
        comm.group_end()
    else:
        comm.group_start()
        for peer in range(world):
            if peer != rank:
                _send(comm, chunks[peer], peer)
                _recv(comm, parts[peer], peer)
        comm.group_end()
    return [chunks[r] if r == rank else parts[r] for r in range(world)]


def reduce_scatter_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """Rows ``[rank * L, (rank + 1) * L)`` of the sum over ranks of ``input_tensor``
    (contiguous, ``world * L`` rows), with the one-shot all-reduce's arithmetic: the
    unreduced exchange, then the rank-order sum (this rank's chunk read in place)."""
    return rank_order_sum(exchange_rank_order(comm, input_tensor, rank, world))


def all_reduce_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """The sum over ranks of ``input_tensor`` at any size, with the one-shot
    all-reduce's arithmetic: its rows (zero-padded to a multiple of ``world``) are
    reduce-scattered in rank order, then all-gathered, which moves bits without
    arithmetic. A row has the same bits as the one-shot path would give it."""
    rows = input_tensor.shape[0]
    flat = input_tensor.reshape(rows, -1)
    padded = -(-rows // world) * world
    if padded != rows or not flat.is_contiguous():
        buffer = flat.new_zeros((padded, flat.shape[1]))
        buffer[:rows].copy_(flat)
    else:
        buffer = flat
    own = reduce_scatter_rank_order(comm, buffer, rank, world)
    gathered = torch.empty_like(buffer)
    comm.all_gather(gathered, own)
    return gathered[:rows].view(input_tensor.shape)


def _send(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.send(tensor, peer)


def _recv(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.recv(tensor, peer)


__all__ = [
    "all_reduce_rank_order",
    "exchange_rank_order",
    "rank_order_sum",
    "rank_order_sum_kernel",
    "reduce_scatter_rank_order",
]
