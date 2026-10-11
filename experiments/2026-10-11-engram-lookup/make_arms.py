#!/usr/bin/env python3
"""Write the B10 Engram lookup arm over r6d: every vLLM file the port's commits change,
taken from the vLLM branch, mounted over the r6d image; and the kernel bundle of its
tests and its bench against B12X's kernel.

usage: make_arms.py [--vllm ~/projects/vllm-ds41-tilelang-migration]
"""
import argparse
import json
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
BASE = ROOT / "experiments/2026-10-10-r6d-deterministic/r6d-tp4.json"
KBENCH = ROOT / "experiments/2026-10-09-tilelang-migration/kbench.py"
R6D = "r6d-candidate"  # the vLLM branch of the r6d image's tree
ARM = "b72a66fa4"  # the Engram lookup switch, on its module with small steps over more blocks
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/test_deepseek_v41_tilelang_engram_lookup.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "lookup"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/lookup/{path}", f"{TREE}/{path}", "ro"] for path in files]
    (HERE / "lookup.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "lookup"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    for name in ("kbench.py", "bench_engram_lookup.py"):
        source = KBENCH if name == "kbench.py" else HERE / name
        shutil.copyfile(source, bundle / name)
        mounts.append([name, f"/b/{name}"])
    run = ("import subprocess, sys; codes = [subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', "
           "'-p', 'no:cacheprovider', '-rfE', " + ", ".join(repr(t) for t in TESTS) + "]), "
           "subprocess.call([sys.executable, '/b/bench_engram_lookup.py'])]; print('exit codes', codes); "
           "sys.exit(max(codes))")
    candidate = {"description": "B10 Engram lookup: bits against B12X's kernel, then time per call against it, "
                                "decode and prefill, warm and cold, mapped-host and device tables.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-c", run], "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/lookup: {len(files)} files ({', '.join(files)}); lookup.json; bundles/lookup")


if __name__ == "__main__":
    main()
