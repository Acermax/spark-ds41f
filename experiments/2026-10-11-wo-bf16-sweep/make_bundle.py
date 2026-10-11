#!/usr/bin/env python3
"""Write a kernel-only bundle of WO-A's W8A16 sweep: the WO module files and tests at a
vLLM commit, the migration's kbench, and a sweep script.

usage: make_bundle.py [--vllm ~/projects/vllm-ds41-tilelang-migration] [--split]

Without --split: the commit without warp specialization and sweep_wo_a.py (bundles/wosweep).
With --split: the split-K commit and sweep_wo_a_splitk.py (bundles/wosplit).
"""
import argparse
import json
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments/2026-10-10-r6d-deterministic/r6d-tp4.json"
KBENCH = ROOT / "experiments/2026-10-09-tilelang-migration/kbench.py"
R6D = "r6d-candidate"
ARM = "145a64d10"  # W8A16 WO-A without warp specialization
SPLIT_ARM = "cec0da371"  # WO-A in four K shards, split over CTAs at decode
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    arm, script, name = ((SPLIT_ARM, "sweep_wo_a_splitk.py", "wosplit") if args.split
                         else (ARM, "sweep_wo_a.py", "wosweep"))
    description = ("WO-A split-K: tests, then split against in-kernel shards over tiles, with bits."
                   if args.split else
                   "WO-A W8A16 without warp specialization: tests, then a tile sweep with bits and replays.")

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, arm, "--", "vllm").decode().split()
    bundle = HERE / "bundles" / name
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        mounted = path.replace("/", "__")
        (bundle / mounted).write_bytes(git("show", f"{arm}:{path}"))
        mounts.append([mounted, f"{TREE}/{path}"])
    for mounted, source in (("kbench.py", KBENCH), (script, HERE / script)):
        shutil.copyfile(source, bundle / mounted)
        mounts.append([mounted, f"/b/{mounted}"])
    run = ("import subprocess, sys; codes = [subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', "
           "'-p', 'no:cacheprovider', '-rfE', " + ", ".join(repr(t) for t in TESTS) + "]), "
           f"subprocess.call([sys.executable, '/b/{script}'])]; print('exit codes', codes); sys.exit(max(codes))")
    image = json.loads(BASE.read_text())["container"]["image"]
    candidate = {"description": description,
                 "image": image, "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-c", run], "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"bundles/{name}: {len(files)} files ({', '.join(files)})")


if __name__ == "__main__":
    main()
