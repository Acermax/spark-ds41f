#!/usr/bin/env python3
"""Localize a temperature-0 difference: per-layer row digests of one prompt served alone
and in token_determinism's mixed15 company, on an arm with VLLM_DS41_ROW_TRACE_DIR.

usage (a lab measure extra): row_trace_session.py BASE_URL [--prompt 11] [--around 7]
        [--trace-dir ~/projects/spark-ds41f/cache/row-trace] [--nodes dgx1,dgx2,dgx3,dgx4]

1. The prompt alone (token ids returned) gives its prompt length and tokens.
2. Every node's spec lists the rows (position, input token) feeding generated tokens
   around --around, then the prompt is traced alone and in mixed15 (fifteen prompts
   at once, the long prompt 0.3 s later), each phase's logs renamed apart.
3. For each traced row: the first label (embed, layer<i>) whose digests differ between
   the phases, with each phase's step row count there. One JSON line per row, then a
   summary. Exits 0 either way.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "token_determinism", os.path.join(HERE, "..", "2026-10-09-tilelang-migration", "token_determinism.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)


def complete_ids(base, model, prompt, tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "ignore_eos": True, "cache_salt": uuid.uuid4().hex, "logprobs": True,
            "return_tokens_as_token_ids": True, "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        reply = json.load(response)
    content = reply["choices"][0]["logprobs"]["content"]
    ids = [int(c["token"].split(":", 1)[1]) for c in content]
    return reply["usage"]["prompt_tokens"], ids


def _shell(node, command, head):
    """``command`` on ``node``; the head (this machine, where lab runs workloads) locally."""
    argv = ["bash", "-c", command] if node == head else ["ssh", "-o", "BatchMode=yes", node, command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


def on_nodes(nodes, command):
    for node in nodes:
        result = _shell(node, command, nodes[0])
        if result.returncode:
            raise SystemExit(f"{node}: {result.stderr.strip()}")


def read_logs(nodes, directory, phase):
    rows = []
    for node in nodes:
        out = _shell(node, f"cat {directory}/{phase}-rank*.jsonl 2>/dev/null", nodes[0]).stdout
        rows += [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--prompt", type=int, default=11)
    parser.add_argument("--around", type=int, default=7)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--trace-dir", default="~/projects/spark-ds41f/cache/row-trace")
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3,dgx4")
    args = parser.parse_args()
    base, nodes, directory = args.base_url.rstrip("/"), args.nodes.split(","), args.trace_dir
    client = td.Client(base)
    prompts = td.PROMPTS + td.MORE_PROMPTS
    prompt = prompts[args.prompt]
    prompt_len, ids = complete_ids(base, client.model, prompt, args.tokens)
    # Row k-1 (input: generated token k-1) produces generated token k.
    lo, hi = max(1, args.around - 3), args.around + 2
    targets = [[prompt_len + k - 1, ids[k - 1]] for k in range(lo, hi + 1)]
    spec = json.dumps({"rows": targets})
    on_nodes(nodes, f"mkdir -p {directory} && echo '{spec}' > {directory}/spec.json.tmp && "
                    f"mv {directory}/spec.json.tmp {directory}/spec.json")
    time.sleep(1)
    long_text = td.long_prompt(700)
    phases = {}
    for phase in ("alone", "mixed"):
        if phase == "alone":
            complete_ids(base, client.model, prompt, args.tokens)
        else:
            td.together(client, [(0.0, p, args.tokens, None) for p in prompts] + [(0.3, long_text, 64, None)])
        on_nodes(nodes, f"cd {directory} && for f in rank*.jsonl; do [ -e \"$f\" ] && mv \"$f\" {phase}-\"$f\"; done; true")
        phases[phase] = read_logs(nodes, directory, phase)
    on_nodes(nodes, f"rm -f {directory}/spec.json")
    summary = []
    for position, token in targets:
        per = {}
        for phase, rows in phases.items():
            mine = [r for r in rows if r["pos"] == position and r["tok"] == token]
            # The last step that computed this row (verification may repeat it).
            last = {}
            for r in mine:
                last[r["label"]] = r
            per[phase] = last
        labels = [label for label in per["alone"] if label in per["mixed"]]
        first = next((label for label in labels
                      if per["alone"][label]["digests"] != per["mixed"][label]["digests"]), None)
        row = {"pos": position, "tok": token, "labels_compared": len(labels), "first_difference": first}
        if first is not None:
            row["alone_rows"] = per["alone"][first]["rows"]
            row["mixed_rows"] = per["mixed"][first]["rows"]
        summary.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"summary": "row trace", "prompt": args.prompt, "prompt_tokens": prompt_len,
                      "rows": summary}), flush=True)


if __name__ == "__main__":
    main()
