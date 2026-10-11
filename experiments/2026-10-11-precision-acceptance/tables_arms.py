#!/usr/bin/env python3
"""DSpark acceptance over sixteen texts for several arms, each paired with the first.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/determinism/RUN/sweep-<arm>.jsonl)

Per arm: the mean over prompts of accepted tokens per draft and of decode tok/s, then
the mean per-prompt change against the first arm with its 95% interval (paired by
prompt, so each text's own predictability cancels). A precision change that moves the
text moves single-prompt acceptance either way; a shift that holds across the sixteen
prompts is the change's own.
"""
import json
import os
import statistics
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
D = f"results/private/determinism/{RUN}/sweep-{{}}.jsonl"


def rows(arm):
    path = D.format(arm)
    if not os.path.exists(path):
        return {}
    lines = [json.loads(x) for x in open(path) if x.strip().startswith("{")]
    return {r["prompt"]: r for r in lines if "prompt" in r}


def paired(values):
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, half


data = {a: rows(a) for a in ARMS}
first = data[ARMS[0]]
print("acceptance sweep: sixteen distinct prompts alone, thinking off, 256 tokens")
for a in ARMS:
    r = data[a]
    if not r:
        print(f"  {a:10s} (no sweep)")
        continue
    acc = statistics.mean(x["accepted_per_draft"] for x in r.values())
    tps = statistics.mean(x["tps"] for x in r.values())
    line = f"  {a:10s} accepted/draft {acc:.3f}   tok/s {tps:6.2f}   prompts {len(r)}"
    common = sorted(set(r) & set(first))
    if a != ARMS[0] and common:
        d_acc, h_acc = paired([r[p]["accepted_per_draft"] - first[p]["accepted_per_draft"] for p in common])
        d_tps, h_tps = paired([100 * (r[p]["tps"] - first[p]["tps"]) / first[p]["tps"] for p in common])
        line += f"   paired: accepted {d_acc:+.3f} ±{h_acc:.3f}, tok/s {d_tps:+.1f}% ±{h_tps:.1f}%"
    print(line)
