#!/usr/bin/env python3
"""Write the MoE FP32-partials arm over r6d: every vLLM file the rank-order all-reduce and
the FP32-partials commits change, taken from the vLLM branch, mounted over the r6d image;
and the kernel bundle of their tests. Its control is the all-reduce arm
(experiments/2026-10-11-allreduce-rank-order/allreduce.json): the same files but the MoE's.

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
ARM = "f48fb9940"  # the MoE FP32-partials switch, on its module and the all-reduce arm
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/test_deepseek_v41_tilelang_collectives.py",
         "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "moefp32"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/moefp32/{path}", f"{TREE}/{path}", "ro"] for path in files]
    (HERE / "moefp32.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "moefp32"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    candidate = {"description": "MoE FP32 partials and the FP32 rank-order sum: their tests, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider", "-rfE", *TESTS],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/moefp32: {len(files)} files ({', '.join(files)}); moefp32.json; bundles/moefp32")


if __name__ == "__main__":
    main()
