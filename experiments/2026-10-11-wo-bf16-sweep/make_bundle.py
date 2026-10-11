#!/usr/bin/env python3
"""Write the kernel-only bundle of WO-A's W8A16 sweep: the WO module files and tests at the
vLLM commit without warp specialization, the migration's kbench, and sweep_wo_a.py.

usage: make_bundle.py [--vllm ~/projects/vllm-ds41-tilelang-migration]
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
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    bundle = HERE / "bundles" / "wosweep"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    for name, source in (("kbench.py", KBENCH), ("sweep_wo_a.py", HERE / "sweep_wo_a.py")):
        shutil.copyfile(source, bundle / name)
        mounts.append([name, f"/b/{name}"])
    run = ("import subprocess, sys; codes = [subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', "
           "'-p', 'no:cacheprovider', '-rfE', " + ", ".join(repr(t) for t in TESTS) + "]), "
           "subprocess.call([sys.executable, '/b/sweep_wo_a.py'])]; print('exit codes', codes); sys.exit(max(codes))")
    image = json.loads(BASE.read_text())["container"]["image"]
    candidate = {"description": "WO-A W8A16 without warp specialization: tests, then a tile sweep with bits and replays.",
                 "image": image, "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-c", run], "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"bundles/wosweep: {len(files)} files ({', '.join(files)})")


if __name__ == "__main__":
    main()
