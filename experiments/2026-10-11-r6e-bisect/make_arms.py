#!/usr/bin/env python3
"""Write the r6e bisect arms: r6d with the r6e stack (vLLM branch r6e-draft2) up to each
piece, every arm without DSpark's adaptive verification, so only r6e's own difference
(prompt 4 in mixed and mixed15) is left to find.

Each arm mounts every vLLM file that commits up to its prefix change, at that prefix, over
the r6d image. The head stays BF16 (r6d's VLLM_DS41_HEAD_DTYPE=model) until the FP32-head
commit, as in the r6e stack.

Window 2's arm: r6e-trace, the full candidate without adaptive verification and without
CUDA graphs, with the debug row trace (vLLM branch r6e-trace). Without a trace spec the
trace does nothing, so the same arm first checks that the difference survives eager.

usage: make_arms.py [--vllm ~/projects/vllm-ds41-tilelang-migration]
"""
import argparse
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
BASE = ROOT / "experiments/2026-10-10-r6d-deterministic/r6d-tp4.json"
R6D = "r6d-candidate"  # the vLLM branch of the r6d image's tree
TREE = "/opt/spark3/candidate/vllm"
TRACE = "a8b3ef9aa"  # r6e-draft2 + the debug row trace, SP parts digested by their sum
# (label, r6e-draft2 commit, FP32 head): each prefix adds one piece.
PREFIXES = [
    ("p1-allreduce", "a0d2d1fd1", False),  # SP WO slices, fused SP sum, rank-order all-reduce
    ("p2-moe", "14d3a5dbe", False),  # MoE: one rounding of each rank's partial
    ("p3-head", "3737e4b72", True),  # FP32 head logits
    ("p4-precision", "bfeb18c37", True),  # FP32 mHC projection, reference-precision SwiGLU
    ("p5-b3", "37c727038", True),  # port B3, cache writers
    ("p6-b4", "26f1b3055", True),  # port B4, compressor
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    base = json.loads(BASE.read_text())
    for label, commit, fp32_head in PREFIXES:
        files = [p for p in git("diff", "--name-only", R6D, commit).decode().split() if p.startswith("vllm/")]
        overlay = HERE / "overlay" / label
        for path in files:
            out = overlay / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(git("show", f"{commit}:{path}"))
        cluster = json.loads(json.dumps(base))
        cluster["container"]["mounts"] += [
            [f"{{home}}/projects/spark-ds41f/{REL}/overlay/{label}/{path}", f"{TREE}/{path}", "ro"]
            for path in files]
        cluster["environment"]["VLLM_DS41_HEAD_DTYPE"] = "float32" if fp32_head else "model"
        at = cluster["serve_args"].index("--speculative-config") + 1
        speculative = json.loads(cluster["serve_args"][at])
        speculative["enable_adaptive_verification"] = False
        speculative.pop("adaptive_verification_cost_scale", None)
        cluster["serve_args"][at] = json.dumps(speculative, separators=(",", ":"))
        (HERE / f"{label}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"{label}: {commit}, {len(files)} files, head {'float32' if fp32_head else 'model'}")
    # Window 2: the full candidate without adaptive verification, eager, then traced.
    fixed = json.loads((ROOT / "experiments/2026-10-11-r6e-candidate/r6e-fixed.json").read_text())
    eager = json.loads(json.dumps(fixed))
    eager["serve_args"] = [*eager["serve_args"], "--enforce-eager"]
    files = [p for p in git("diff", "--name-only", R6D, TRACE).decode().split() if p.startswith("vllm/")]
    for path in files:
        out = HERE / "overlay" / "trace" / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{TRACE}:{path}"))
    trace = json.loads(json.dumps(eager))
    trace["container"]["mounts"] = [m for m in trace["container"]["mounts"] if "/overlay/r6e/" not in m[0]]
    trace["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/trace/{path}", f"{TREE}/{path}", "ro"] for path in files]
    trace["environment"]["VLLM_DS41_ROW_TRACE_DIR"] = "/cache/row-trace"
    (HERE / "r6e-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(f"r6e-trace: {TRACE}, {len(files)} files, eager")


if __name__ == "__main__":
    main()
