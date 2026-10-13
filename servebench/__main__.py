"""servebench command line: ``run`` measures a live server, ``compare`` judges two runs."""

import argparse
import datetime
import json
import os
import sys
import urllib.request

from . import SCHEMA, __version__, engine, load, prompts, report


def model_id(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def command_run(args: argparse.Namespace) -> int:
    prompt_set = prompts.load(args.prompts)
    spec = load.Load(mode=args.mode, concurrency=args.concurrency, rate=args.rate,
                     warmup_s=args.warmup, duration_s=args.duration, batches=args.batches,
                     max_tokens=args.max_tokens, fixed_length=not args.natural_length,
                     seed=args.seed)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    model = args.model or model_id(args.url)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    raw = load.run(args.url, model, prompt_set, spec, vllm=not args.no_vllm, extra_body=extra)
    summary = report.summarize(raw, slo_ttft_s=args.slo_ttft, slo_tpot_ms=args.slo_tpot)
    result = {
        "schema": SCHEMA, "tool": {"name": "servebench", "version": __version__},
        "started_utc": started, "label": args.label,
        "server": {"url": args.url, "model": model, "adapter": "openai" if args.no_vllm else "vllm"},
        "workload": spec.describe() | {"prompts": args.prompts, "prompt_count": len(prompt_set),
                                       "prompts_sha256": prompts.digest(prompt_set),
                                       "extra_body": extra},
        "summary": summary,
    }
    if args.save_requests:
        origin = raw["start"]
        result["requests"] = [
            {key: (round(value - origin, 4) if key in ("sent", "first", "end") and value is not None else value)
             for key, value in record.items() if key != "events"}
            | {"events": [[round(t - origin, 4), n] for t, n in record["events"]]}
            for record in raw["records"]]
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    tps = summary.get("output_tps", {})
    line = {"summary": f"servebench {spec.mode}", "label": args.label, "mode": spec.mode,
            "concurrency": spec.concurrency if spec.mode != "poisson" else None,
            "rate": spec.rate if spec.mode == "poisson" else None,
            "tps_mean": tps.get("mean"), "ci95_pct": tps.get("ci95_pct"), "windows": tps.get("n"),
            "steps_per_s": summary.get("server", {}).get("steps_per_s", {}).get("mean"),
            "tokens_per_step": summary.get("server", {}).get("output_tokens_per_step", {}).get("mean"),
            "ttft_p50_s": summary["latency"]["ttft_s"].get("p50"),
            "tpot_p50_ms": summary["latency"]["tpot_ms"].get("p50"),
            "failed": summary["requests"]["failed"], "output": args.output}
    print(json.dumps(line), flush=True)
    return 1 if summary["requests"]["failed"] else 0


def command_engine(args: argparse.Namespace) -> int:
    if args.grid == "default":
        longest = args.max_context or engine.max_model_len(args.url)
        shapes = engine.default_shapes(drafts=args.drafts, max_context=longest)
    else:
        with open(args.grid, encoding="utf-8") as handle:
            shapes = json.load(handle)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    raw = engine.run(args.url, shapes, replays=args.replays, warmup=args.warmup)
    rows = engine.summarize(raw)
    result = {"schema": engine.SCHEMA, "tool": {"name": "servebench", "version": __version__},
              "started_utc": started, "label": args.label,
              "server": {"url": args.url, "workers": raw["workers"], "rank": raw["rank"],
                         "prefix_cache_reset": raw["prefix_cache_reset"]},
              "workload": {"grid": args.grid, "replays": args.replays, "warmup": args.warmup,
                           "drafts": args.drafts},
              "engine": rows}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    for row in rows:
        d = row["step_ms"]
        print(f"{row['label']:32s} {d['mean']:9.3f} ms ±{d['ci95_pct'] or 0:.2f}%", file=sys.stderr)
    widest = max((row["step_ms"].get("ci95_pct") or 0) for row in rows)
    print(json.dumps({"summary": "servebench engine", "label": args.label, "shapes": len(rows),
                      "widest_ci95_pct": widest, "prefix_cache_reset": raw["prefix_cache_reset"],
                      "output": args.output}), flush=True)
    return 0 if raw["prefix_cache_reset"] else 1


def command_compare(args: argparse.Namespace) -> int:
    runs = []
    for path in (args.base, args.candidate):
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    if runs[0].get("schema") != runs[1].get("schema"):
        raise SystemExit(f"different result kinds: {runs[0].get('schema')} vs {runs[1].get('schema')}")
    if runs[0].get("schema") == engine.SCHEMA:
        result = engine.compare(runs[0], runs[1], tolerance_pct=args.tolerance)
    else:
        result = report.compare(runs[0], runs[1], tolerance_pct=args.tolerance)
    labels = [run.get("label") or path for run, path in zip(runs, (args.base, args.candidate))]
    print(report.format_comparison(result, labels[0][:12], labels[1][:12]))
    if args.json:
        print(json.dumps(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="servebench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="measure a live OpenAI-compatible server")
    run.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    run.add_argument("--mode", choices=("closed", "poisson", "burst"), default="closed")
    run.add_argument("--concurrency", type=int, default=8, help="closed and burst: requests in flight")
    run.add_argument("--rate", type=float, default=1.0, help="poisson: requests per second")
    run.add_argument("--warmup", type=float, default=60.0, help="seconds before the window")
    run.add_argument("--duration", type=float, default=300.0, help="window length in seconds")
    run.add_argument("--batches", type=int, default=10, help="sub-windows for the intervals")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--natural-length", action="store_true",
                     help="let requests stop at end of sequence (default: exactly --max-tokens each)")
    run.add_argument("--prompts", default="default", help="prompt set name or JSONL path")
    run.add_argument("--seed", type=int, default=0, help="orders the prompts and the arrivals")
    run.add_argument("--model", help="model id (default: the server's first)")
    run.add_argument("--extra-body", help="JSON merged into every request body")
    run.add_argument("--no-vllm", action="store_true",
                     help="plain OpenAI requests (no cache_salt, ignore_eos, per-chunk usage)")
    run.add_argument("--slo-ttft", type=float, help="goodput: time to first token limit, seconds")
    run.add_argument("--slo-tpot", type=float, help="goodput: time per output token limit, ms")
    run.add_argument("--label", help="name for tables")
    run.add_argument("--output", help="write the full result JSON here")
    run.add_argument("--save-requests", action="store_true", help="keep every request's timeline")
    run.set_defaults(func=command_run)
    grid = sub.add_parser("engine", help="time identical dummy steps inside the server (engine cost grid)")
    grid.add_argument("url", help="base URL of a server with the dev-mode collective RPC")
    grid.add_argument("--grid", default="default", help="'default' or a JSON file of shapes")
    grid.add_argument("--replays", type=int, default=30, help="timed rounds over every shape")
    grid.add_argument("--warmup", type=int, default=2, help="untimed runs per shape first")
    grid.add_argument("--drafts", type=int, default=5, help="drafts per request in verify rows")
    grid.add_argument("--max-context", type=int, help="longest context (default: the server's)")
    grid.add_argument("--label", help="name for tables")
    grid.add_argument("--output", help="write the result JSON here")
    grid.set_defaults(func=command_engine)
    compare = sub.add_parser("compare", help="judge a candidate run against a base run")
    compare.add_argument("base")
    compare.add_argument("candidate")
    compare.add_argument("--tolerance", type=float, default=3.0,
                         help="budget in %% for metrics without an interval")
    compare.add_argument("--json", action="store_true", help="also print the comparison as JSON")
    compare.set_defaults(func=command_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
