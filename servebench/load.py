"""Load generation.

- ``closed``: N workers each keep one request in flight, starting the next as soon as
  theirs ends, so the server sees a steady N with new prompts' prefills mixed into decode
  steps, as it does under real traffic.
- ``poisson``: requests arrive as a Poisson process at a fixed rate whatever the server
  does (open loop); queueing shows up as latency.
- ``burst``: N requests at once and nothing else, the old distinct-stream shape, kept for
  comparison with earlier numbers.

Closed and Poisson runs warm up first, then measure for a fixed duration split into
equal sub-windows (batch means: each sub-window is one sample for the intervals), then
stop issuing and let the in-flight requests drain outside the window. Server counters
are read at every sub-window boundary.
"""

import itertools
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass

from . import client, prom


@dataclass
class Load:
    mode: str = "closed"          # closed | poisson | burst
    concurrency: int = 8          # closed and burst
    rate: float = 1.0             # poisson: requests per second
    warmup_s: float = 60.0
    duration_s: float = 300.0
    batches: int = 10
    max_tokens: int = 256
    fixed_length: bool = True     # every request writes exactly max_tokens (vLLM adapter)
    seed: int = 0
    max_inflight: int = 512       # poisson safety cap; arrivals past it are counted, not sent

    def describe(self) -> dict:
        return asdict(self)


def payload(model: str, prompt: dict, load: Load, *, vllm: bool, extra_body: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt["text"]}],
            "max_tokens": load.max_tokens, "temperature": 0}
    if vllm:
        # A fresh salt per request: a prompt seen earlier in the run must not hit the
        # prefix cache, or repeats would skip their prefill.
        body["cache_salt"] = uuid.uuid4().hex
        if load.fixed_length:
            body["ignore_eos"] = True
            body["min_tokens"] = load.max_tokens
    body.update(extra_body)
    return body


def _sleep_until(moment: float, clock) -> None:
    while True:
        remaining = moment - clock()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


def run(base_url: str, model: str, prompts: list[dict], load: Load, *, vllm: bool = True,
        extra_body: dict | None = None, metrics: dict | None = None, clock=time.monotonic,
        log=lambda message: print(message, file=sys.stderr, flush=True)) -> dict:
    """Run ``load`` and return the raw run: records, window bounds, boundary scrapes."""
    extra_body = extra_body or {}
    metrics = prom.VLLM if metrics is None else metrics
    order = list(prompts)
    random.Random(load.seed).shuffle(order)
    counter = itertools.count()
    lock = threading.Lock()
    records: list[dict] = []
    rejected = [0]

    def send(slot: int) -> None:
        index = next(counter)
        prompt = order[index % len(order)]
        body = payload(model, prompt, load, vllm=vllm, extra_body=extra_body)
        record = client.stream_chat(base_url, body, continuous_usage=vllm, clock=clock)
        record.update(prompt_id=prompt["id"], slot=slot, index=index)
        with lock:
            records.append(record)

    start = clock()
    if load.mode == "burst":
        threads = [threading.Thread(target=send, args=(slot,)) for slot in range(load.concurrency)]
        before = prom.scrape(base_url, metrics)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        after = prom.scrape(base_url, metrics)
        return {"records": records, "start": start, "window": None,
                "boundaries": [], "scrapes": [before, after], "scrape_times": [start, clock()],
                "rejected": 0}

    w0 = start + load.warmup_s
    w1 = w0 + load.duration_s
    boundaries = [w0 + k * load.duration_s / load.batches for k in range(load.batches + 1)]
    scrapes: list = [None] * len(boundaries)
    scrape_times: list = [None] * len(boundaries)

    def scraper() -> None:
        for k, boundary in enumerate(boundaries):
            _sleep_until(boundary, clock)
            scrape_times[k] = clock()
            scrapes[k] = prom.scrape(base_url, metrics)
            if k:
                log(f"servebench: window {k}/{load.batches} done")

    scraping = threading.Thread(target=scraper, daemon=True)
    scraping.start()
    log(f"servebench: {load.mode} warm-up {load.warmup_s:.0f} s, then {load.duration_s:.0f} s "
        f"in {load.batches} windows")
    workers: list[threading.Thread] = []
    if load.mode == "closed":
        def worker(slot: int) -> None:
            while clock() < w1:
                send(slot)
        workers = [threading.Thread(target=worker, args=(slot,), daemon=True)
                   for slot in range(load.concurrency)]
        for thread in workers:
            thread.start()
    elif load.mode == "poisson":
        arrivals = random.Random(load.seed + 1)
        moment = start
        while True:
            moment += arrivals.expovariate(load.rate)
            if moment >= w1:
                break
            _sleep_until(moment, clock)
            with lock:
                inflight = sum(1 for thread in workers if thread.is_alive())
            if inflight >= load.max_inflight:
                rejected[0] += 1
                continue
            thread = threading.Thread(target=send, args=(len(workers),), daemon=True)
            thread.start()
            workers.append(thread)
    else:
        raise ValueError(f"unknown mode {load.mode!r}")
    for thread in workers:
        thread.join()
    scraping.join()
    return {"records": records, "start": start, "window": [w0, w1], "boundaries": boundaries,
            "scrapes": scrapes, "scrape_times": scrape_times, "rejected": rejected[0]}
