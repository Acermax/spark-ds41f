#!/usr/bin/env python3
"""The MoE output's cross-rank reduction in FP32 against BF16 on the fabric: the cost of
keeping the partials unrounded, and the FP32 results bit for bit, model-free, one rank
per node.

usage (a lab "fabric" job; the runner adds the probe's rank arguments):
    fabric_moe_reduce.py --rank R --world-size N --master-addr A --master-port P
        [--rows 1 16 ...] [--scatter-rows 2048 ...] [--iters 20]

All-reduce sizes are decode and CED steps (rows of 5120): the communicator's
all_reduce on BF16 partials (today's MoE output) and on FP32 partials (the arm's),
the FP32 one followed by the layer's rounding to BF16. FP32 halves the rows the
one-shot path takes (102 instead of 204); past it the rank-order all-reduce runs.
Reduce-scatter sizes are prefill sequence-parallel chunks. Every FP32 result must
equal the rank-order FP32 sum of the gathered partials, bit for bit, on every rank.
Rank 0 prints one JSON line per size, then a summary; exits 1 on any difference.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta

HIDDEN = 5120
MOE_LAYERS = 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+",
                        default=[1, 6, 16, 48, 96, 102, 103, 150, 204, 205, 300])
    parser.add_argument("--scatter-rows", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if "SPARKNET_ROCE_TOPOLOGY" not in os.environ:
        raise RuntimeError("the arm's environment must select sparknet's one-shot collectives")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=args.world_size, rank=args.rank, local_rank=0,
        distributed_init_method=f"tcp://{args.master_addr}:{args.master_port}",
        timeout=timedelta(seconds=120),
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=args.world_size)
    group = get_tp_group()
    comm = group.device_communicator
    oneshot = comm.b12x_ar_comm
    if oneshot is None or oneshot.disabled:
        raise RuntimeError("sparknet's one-shot all-reduce is required")

    world, rank = args.world_size, args.rank

    def slowest(value: float) -> float:
        t = torch.tensor([value], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return round(float(t.item()), 1)

    def timed(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            dist.barrier(group.cpu_group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)
        return slowest(statistics.median(times))

    def rank_order(x):
        """Every rank's partial, added in FP32 in rank order: the reference sum."""
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.contiguous(), group=group.device_group)
        total = parts[0].to(torch.float32, copy=True)
        for part in parts[1:]:
            total += part.float()
        return total

    def differ(a, b) -> int:
        return int((a.view(torch.int32) != b.view(torch.int32)).sum())

    def partials(rows: int, seed: int):
        gen = torch.Generator(device=device).manual_seed(seed + rank)
        x32 = torch.randn((rows, HIDDEN), generator=gen, device=device) * 4
        return x32, x32.bfloat16()

    failures, deltas = [], {}
    for rows in args.rows:
        x32, x16 = partials(rows, 3000)
        got = comm.all_reduce(x32.clone())
        want = rank_order(x32)
        torch.cuda.synchronize()
        row = {"rows": rows, "fp32_mib": round(rows * HIDDEN * 4 / 2**20, 2),
               "fp32_path": "one-shot" if oneshot.should_custom_ar(x32) else "rank-order",
               "fp32_elements_differ": slowest(differ(got, want))}
        if row["fp32_elements_differ"]:
            failures.append(f"all-reduce {rows} rows: FP32 bits differ from the rank-order sum")
        row["bf16_us"] = timed(lambda x=x16: comm.all_reduce(x.clone()))
        row["fp32_us"] = timed(lambda x=x32: comm.all_reduce(x.clone()))
        row["fp32_round_us"] = timed(lambda x=x32: comm.all_reduce(x.clone()).bfloat16())
        row["delta_us"] = round(row["fp32_round_us"] - row["bf16_us"], 1)
        row["per_step_ms"] = round(MOE_LAYERS * row["delta_us"] / 1000, 3)
        deltas[rows] = row["per_step_ms"]
        if rank == 0:
            print(json.dumps(row), flush=True)
    for rows in args.scatter_rows:
        rows -= rows % world
        x32, x16 = partials(rows, 4000)
        got = comm.reduce_scatter(x32.clone(), dim=0)
        want = rank_order(x32).chunk(world)[rank]
        torch.cuda.synchronize()
        row = {"scatter_rows": rows, "fp32_mib": round(rows * HIDDEN * 4 / 2**20, 2),
               "fp32_elements_differ": slowest(differ(got.contiguous(), want.contiguous()))}
        if row["fp32_elements_differ"]:
            failures.append(f"reduce-scatter {rows} rows: FP32 bits differ from the rank-order sum")
        row["bf16_us"] = timed(lambda x=x16: comm.reduce_scatter(x.clone(), dim=0))
        row["fp32_round_us"] = timed(lambda x=x32: comm.reduce_scatter(x.clone(), dim=0).bfloat16())
        row["delta_us"] = round(row["fp32_round_us"] - row["bf16_us"], 1)
        row["per_chunk_ms"] = round(MOE_LAYERS * row["delta_us"] / 1000, 3)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if rank == 0:
        print(json.dumps({"summary": "MoE reduction, FP32 against BF16, on the fabric",
                          "per_step_ms_by_rows": deltas, "failures": failures}), flush=True)
    dist.barrier(group.cpu_group)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
