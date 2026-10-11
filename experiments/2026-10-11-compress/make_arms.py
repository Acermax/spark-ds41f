#!/usr/bin/env python3
"""Write the B4 compression arm over r6d: every vLLM file the port's commits change,
taken from the vLLM branch, mounted over the r6d image; and the kernel bundle of its
tests and its bench against B12X's compressor path.

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
ARM = "49e598640"  # the compression switch, on its module
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/test_deepseek_v41_tilelang_compress.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "compress"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/compress/{path}", f"{TREE}/{path}", "ro"] for path in files]
    (HERE / "compress.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "compress"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    for name in ("kbench.py", "bench_compress.py"):
        source = KBENCH if name == "kbench.py" else HERE / name
        shutil.copyfile(source, bundle / name)
        mounts.append([name, f"/b/{name}"])
    run = ("import subprocess, sys; codes = [subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', "
           "'-p', 'no:cacheprovider', '-rfE', " + ", ".join(repr(t) for t in TESTS) + "]), "
           "subprocess.call([sys.executable, '/b/bench_compress.py'])]; print('exit codes', codes); "
           "sys.exit(max(codes))")
    candidate = {"description": "B4 compression: accuracy against DeepSeek's arithmetic and B12X, then time per step "
                                "against B12X's path, decode and prefill, warm and cold, ratio 1 and 2.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-c", run], "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/compress: {len(files)} files ({', '.join(files)}); compress.json; bundles/compress")


if __name__ == "__main__":
    main()
