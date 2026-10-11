#!/usr/bin/env python3
"""Write the WO arm over r6d: every vLLM file the BF16 WO-A module and the B1 switch
changes, taken from the vLLM branch, mounted over the r6d image;
and the kernel bundle of their tests.

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
ARM = "50f6f886d"  # B1 switch on the W8A16 WO-A module: per-tile scales, no warp specialization, swept tiles
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py"]
BENCH = ROOT / "experiments/2026-10-09-tilelang-migration/bundles/wo-projection"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "wobf16"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/wobf16/{path}", f"{TREE}/{path}", "ro"] for path in files]
    (HERE / "wobf16.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "wobf16"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    for name in ("kbench.py", "bench_wo.py"):  # B1's bench against B12X's fused WO
        (bundle / name).write_bytes((BENCH / name).read_bytes())
        mounts.append([name, f"/b/{name}"])
    candidate = {"description": "WO with BF16 WO-A against B12X: tests and the B1 bench, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-c", "import subprocess, sys; codes = [subprocess.call([sys.executable, '-m', 'pytest', '-q', "
                          "'--noconftest', '-p', 'no:cacheprovider', '-rfE', " + ", ".join(repr(t) for t in TESTS)
                          + "]), subprocess.call([sys.executable, '/b/bench_wo.py'])]; print('exit codes', codes); "
                          "sys.exit(max(codes))"],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/wobf16: {len(files)} files ({', '.join(files)}); wobf16.json; bundles/wobf16")


if __name__ == "__main__":
    main()
