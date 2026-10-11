#!/usr/bin/env python3
"""Write the r6e-fold arm: the r6e candidate (vLLM branch r6e-draft2) plus the MoE combine
adding the shared expert's output when it runs first (vLLM branch r6e-moe-fold), mounted
over the r6d image with the FP32 head as r6e; and the kernel bundle of the MoE tests.

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
ARM = "7ec14fee1"  # r6e-draft2 + the shared expert's output folded into the combine
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = [p for p in git("diff", "--name-only", R6D, ARM).decode().split() if p.startswith("vllm/")]
    overlay = HERE / "overlay" / "fold"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/fold/{path}", f"{TREE}/{path}", "ro"] for path in files]
    cluster["environment"]["VLLM_DS41_HEAD_DTYPE"] = "float32"
    (HERE / "r6e-fold.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "fold"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    candidate = {"description": "The MoE tests with the shared output folded into the combine, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider", "-rfE", *TESTS],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/fold: {len(files)} files; r6e-fold.json; bundles/fold")


if __name__ == "__main__":
    main()
