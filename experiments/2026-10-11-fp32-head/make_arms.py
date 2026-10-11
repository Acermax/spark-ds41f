#!/usr/bin/env python3
"""Write the FP32-head arm over r6d: every vLLM file the FP32 vocabulary-head commit
changes, taken from the vLLM branch, mounted over the r6d image, with VLLM_DS41_HEAD_DTYPE=float32;
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
ARM = "93eb6686c"  # TileLang vocabulary heads write FP32 logits for an FP32 head dtype
TREE = "/opt/spark3/candidate/vllm"
TESTS = ["tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, ARM, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "fp32head"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ARM}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/fp32head/{path}", f"{TREE}/{path}", "ro"] for path in files]
    cluster["environment"]["VLLM_DS41_HEAD_DTYPE"] = "float32"  # DeepSeek's reference head
    (HERE / "fp32head.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "fp32head"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, *TESTS]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{ARM}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    candidate = {"description": "FP32 vocabulary heads: their tests, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider", "-rfE", *TESTS],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/fp32head: {len(files)} files ({', '.join(files)}); fp32head.json; bundles/fp32head")


if __name__ == "__main__":
    main()
