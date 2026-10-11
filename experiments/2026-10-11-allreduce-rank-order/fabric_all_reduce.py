#!/usr/bin/env python3
"""All-reduces past sparknet's one-shot limit on the fabric: the rank-order all-reduce
(rank-order reduce-scatter, then an exact all-gather) against the one-shot all-reduce
of the same rows in chunks it accepts, bit for bit, model-free, one rank per node.

usage (a lab "fabric" job; the runner adds the probe's rank arguments):
    fabric_all_reduce.py --rank R --world-size N --master-addr A --master-port P
        [--rows 205 300 ...] [--iters 20]

For each size (BF16 rows of 5120): the communicator's all_reduce (with the arm's
switch, the rank-order path past 2 MiB) and the rank-order all-reduce called
directly must both equal the one-shot all-reduce applied to chunks of at most
204 rows, in every element on every rank. NCCL's ring all-reduce is timed beside
them and its differing elements counted. Rank 0 prints one JSON line per size,
then a summary; exits 1 if any rank-order result differs.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta

HIDDEN = 5120
ONESHOT_ROWS = 204  # 204 x 5120 BF16 = 2,088,960 bytes, within the 2 MiB dispatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[205, 224, 300, 512, 1024, 4096, 8192])
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
    nccl = comm.pynccl_comm
    oneshot = comm.b12x_ar_comm
    if nccl is None or nccl.disabled or oneshot is None or oneshot.disabled:
        raise RuntimeError("PyNCCL and sparknet's one-shot all-reduce are both required")
    from vllm.models.deepseek_v4_1.tilelang.collectives import all_reduce_rank_order

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

    def differ(a, b) -> int:
        return int((a.view(torch.int16) != b.view(torch.int16)).sum())

    failures = []
    for rows in args.rows:
        gen = torch.Generator(device=device).manual_seed(2000 + rank)
        x = (torch.randn((rows, HIDDEN), generator=gen, device=device) * 4).bfloat16()
        chunks = []
        for start in range(0, rows, ONESHOT_ROWS):
            chunk = x[start:start + ONESHOT_ROWS].contiguous()
            if not oneshot.should_custom_ar(chunk):
                raise RuntimeError(f"one-shot refused a {chunk.shape[0]}-row chunk")
            chunks.append(oneshot.custom_all_reduce(chunk))
        reference = torch.cat(chunks)
        direct = all_reduce_rank_order(nccl, x, rank, world)
        through = comm.all_reduce(x.clone())
        ring = nccl.all_reduce(x.clone())
        torch.cuda.synchronize()
        row = {"rows": rows, "mib": round(rows * HIDDEN * 2 / 2**20, 2),
               "direct_elements_differ": slowest(differ(direct, reference)),
               "communicator_elements_differ": slowest(differ(through, reference)),
               "nccl_elements_differ": slowest(differ(ring, reference))}
        if row["direct_elements_differ"] or row["communicator_elements_differ"]:
            failures.append(f"{rows} rows: bits differ from the one-shot arithmetic")
        row["nccl_us"] = timed(lambda x=x: nccl.all_reduce(x.clone()))
        row["rank_order_us"] = timed(lambda x=x: all_reduce_rank_order(nccl, x, rank, world))
        row["ratio"] = round(row["rank_order_us"] / row["nccl_us"], 3)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if rank == 0:
        print(json.dumps({"summary": "rank-order all-reduce on the fabric", "failures": failures}), flush=True)
    dist.barrier(group.cpu_group)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
