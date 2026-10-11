#!/usr/bin/env python3
"""Write the r6e candidate arm over r6d: every vLLM file the r6e stack (vLLM branch
r6e-draft: overlap, fused SP sum, rank-order all-reduce, MoE FP32 partials, FP32 head,
FP32 mHC, reference-precision shared SwiGLU, BF16 WO-A with port B1, port B3) changes,
mounted over the r6d image, with VLLM_DS41_HEAD_DTYPE=float32 for the FP32 head; and
the kernel bundle of every DS4.1 TileLang test it touches.

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
ARM = "1e9ac8c16"  # r6e-draft2: the stack with B4, T1 and MoE one rounding; no WO-BF16, no B10
TREE = "/opt/spark3/candidate/vllm"
EXTRA_TESTS = ["tests/kernels/test_deepseek_v41_tilelang_cache_writer.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    changed = git("diff", "--name-only", R6D, ARM).decode().split()
    files = [p for p in changed if p.startswith("vllm/")]
    tests = sorted({p for p in changed if p.startswith("tests/")} | set(EXTRA_TESTS))
    overlay = HERE / "overlay" / "r6e"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/r6e/{path}", f"{TREE}/{path}", "ro"] for path in files]
    cluster["environment"]["VLLM_DS41_HEAD_DTYPE"] = "float32"
    (HERE / "r6e.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "r6e"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *tests]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    candidate = {"description": "The r6e stack's DS4.1 TileLang tests, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider", "-rfE", *tests],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/r6e: {len(files)} files; {len(tests)} test files; r6e.json; bundles/r6e")


if __name__ == "__main__":
    main()
