"""Port B4: one compressor's step, B12X's path (ratio 2: vLLM's Triton carry load and
ring save, then mla_compress over the batch capacity; ratio 1: mla_compress) against the
two TileLang launches sized by the step, at TP4's capacity (8192 tokens, 16 requests).

1. Accuracy: emitted flags and slots equal B12X's; latents against an FP64 recompute of
   DeepSeek's arithmetic, max and RMS error no worse than B12X's.
2. Time per step under CUDA graphs, warm and cold, at decode steps and prefill chunks.
3. Repeatable: graph replays give the eager call's bits.

Exits 1 on a flag or slot difference, a worse error, a replay difference, or TileLang
slower than B12X by more than the 3% budget at any step.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DEVICE, errors, no_worse, per_call, repeatable, slower, timing_line  # noqa: E402

from b12x.attention import mla_compress  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.compressor import _load_partial_state, _save_partial_states  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import compress as tl  # noqa: E402

R, PAGE, BLOCKS, DIM = 16, 8, 64, 512
STEPS = {  # name: tokens per request
    "c1 x1": [1], "c1 x6": [6], "c8 x6": [6] * 8, "c16 x6": [6] * 16, "c16 mixed": [1, 6, 3, 2] * 4,
    "prefill 512": [512], "prefill 2048": [2048], "prefill 8192 / 4": [2048] * 4,
}
gen = torch.Generator(device=DEVICE).manual_seed(17)
failures = []


def buf(shape, dtype=torch.float32, fill=None):
    t = torch.empty(shape, dtype=dtype, device=DEVICE)
    return t if fill is None else t.fill_(fill)


meta = dict(starts=buf((R + 1,), torch.int32), positions=buf((R,), torch.int64),
            slots=buf((CAPACITY,), torch.int64), state_slots=buf((CAPACITY,), torch.int64),
            counts=buf((2,), torch.int32), table=buf((R, 4), torch.int32), row_ids=torch.arange(R, device=DEVICE))
weight = torch.rand(DIM, generator=gen, device=DEVICE) + 0.5


def load_step(lengths):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    nt, nr = starts[-1], len(lengths)
    s = torch.full((R + 1,), nt, dtype=torch.int32)
    s[: nr + 1] = torch.tensor(starts, dtype=torch.int32)
    meta["starts"].copy_(s)
    pos = torch.randint(1, 4000, (R,), generator=gen, device=DEVICE)
    pos[nr:] = -1
    meta["positions"].copy_(pos)
    meta["table"].copy_(torch.randint(1, BLOCKS, (R, 4), generator=gen, device=DEVICE).to(torch.int32))
    meta["slots"].copy_(torch.randint(0, 10**6, (CAPACITY,), generator=gen, device=DEVICE))
    ss = torch.full((CAPACITY,), -1, dtype=torch.int64, device=DEVICE)
    for r, n in enumerate(lengths):
        p = pos[r] + torch.arange(n, device=DEVICE)
        ss[starts[r]: starts[r] + n] = meta["table"][r, 0].long() * PAGE + p % PAGE
    meta["state_slots"].copy_(ss)
    meta["counts"].copy_(torch.tensor([nt, nr], dtype=torch.int32))
    return nt, starts


def reference64(values, gates, starts, ratio, carry_v, carry_g, tags, emitted):
    """FP64 latents of the emitted tokens (DeepSeek's arithmetic, rounding where it rounds)."""
    out = torch.zeros(values.shape[0], DIM, dtype=torch.float64, device=DEVICE)
    for r in range(len(starts) - 1):
        for token in range(starts[r], starts[r + 1]):
            if not emitted[token]:
                continue
            if ratio == 1:
                x = values[token].double()
            else:
                first = token == starts[r]
                v0 = (carry_v[r] if first else values[token - 1]).double()
                g0 = (carry_g[r] if first else gates[token - 1]).double()
                g = torch.stack((g0, gates[token].double())).softmax(0)
                x = (v0 * g[0] + values[token].double() * g[1]).to(torch.bfloat16).double()
            out[token] = x * torch.rsqrt(x.square().mean() + 1e-20) * weight.double()
    return out


for ratio in (1, 2):
    print(f"ratio {ratio}")
    vdtype = torch.float32 if ratio == 2 else torch.bfloat16
    values = (torch.randn(CAPACITY, DIM, generator=gen, device=DEVICE) * 3).to(vdtype)
    gates = torch.randn(CAPACITY, DIM, generator=gen, device=DEVICE) * 2
    pool0 = torch.randn(BLOCKS, PAGE * 2 * DIM, generator=gen, device=DEVICE)
    side = {k: [buf((CAPACITY, DIM), torch.bfloat16), buf((CAPACITY,), torch.bool), buf((CAPACITY,), torch.int64),
                buf((R, DIM)), buf((R, DIM)), buf((R,), torch.int64), buf((R,), torch.int64), pool0.clone()]
            for k in ("b12x", "tilelang")}
    out_b, em_b, sl_b, pv_b, pg_b, tag_b, ids_b, pool_b = side["b12x"]
    out_p, em_p, sl_p, pv_p, pg_p, tag_p, ids_p, pool_p = side["tilelang"]
    plan = mla_compress.plan(mla_compress.Caps(device=DEVICE, max_tokens=CAPACITY, max_requests=R,
                                               max_states=R, ratio=ratio))
    state_ids_b = ids_b if ratio == 2 else meta["row_ids"]
    kwargs = dict(values=values, weight=weight, query_start_loc=meta["starts"], positions=meta["positions"],
                  state_ids=state_ids_b, destination_slots=meta["slots"], live_counts=meta["counts"],
                  out=out_b, emitted=em_b, emitted_slots=sl_b)
    if ratio == 2:
        kwargs.update(gates=gates, pending_values=pv_b, pending_gates=pg_b, pending_position=tag_b)
    meta["counts"].zero_()  # the primer call compresses nothing

    def prepare(state):
        primer = state.bind(**kwargs)
        return PreparedCall(run=lambda: state.run(primer))

    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name=f"compress/{ratio}", prepare_call=prepare),))
    binding = mla_compress.bind(plan, **kwargs)
    tl.placeholder(R, torch.device(DEVICE))

    for name, lengths in STEPS.items():
        nt, starts = load_step(lengths)
        rows = nt

        def base():
            if ratio == 2:
                _load_partial_state[(R,)](pool_b, meta["table"], meta["starts"], meta["positions"], pv_b, pg_b,
                                          tag_b, ids_b, meta["counts"], pool_b.stride(0), meta["table"].stride(0), PAGE)
                _save_partial_states[(CAPACITY,)](pool_b, meta["state_slots"], values, gates, meta["counts"],
                                                  pool_b.stride(0), PAGE)
            mla_compress.run(binding)

        def port():
            if ratio == 2:
                tl.gather_partial_state(pool_p, meta["table"], meta["starts"], meta["positions"], meta["counts"],
                                        pv_p, pg_p, tag_p, ids_p, page=PAGE)
                tl.compress(rows, 2, values, gates, weight, meta["starts"], meta["positions"], ids_p, meta["slots"],
                            meta["counts"], out_p, em_p, sl_p, pending_values=pv_p, pending_gates=pg_p, tags=tag_p,
                            pool=pool_p, state_slots=meta["state_slots"], page=PAGE)
            else:
                tl.compress(rows, 1, values, None, weight, meta["starts"], meta["positions"], meta["row_ids"],
                            meta["slots"], meta["counts"], out_p, em_p, sl_p)

        base()
        port()
        torch.cuda.synchronize()
        if not (torch.equal(em_p[:rows], em_b[:rows]) and torch.equal(sl_p[:rows], sl_b[:rows])):
            failures.append(f"ratio {ratio} {name}: emitted flags or slots differ")
        ref = reference64(values, gates, starts, ratio, pv_p, pg_p, tag_p, em_b[:rows].tolist() + [False])
        e_b, e_p = errors(out_b[:rows], ref[:rows]), errors(out_p[:rows], ref[:rows])
        if not no_worse(e_p, e_b):
            failures.append(f"ratio {ratio} {name}: error {e_p} against B12X's {e_b}")
        b, p = per_call(base, rows), per_call(port, rows)
        print(timing_line(rows, b, p) + f"  {name}  err max {e_b[0]:.2e} / {e_p[0]:.2e}")
        if slower(b, p):
            failures.append(f"ratio {ratio} {name}: slower ({p[0]:.2f}/{p[1]:.2f} us against {b[0]:.2f}/{b[1]:.2f})")
        if (bad := repeatable(port, lambda: [out_p[:rows], em_p[:rows], sl_p[:rows]])):
            failures.append(f"ratio {ratio} {name}: {bad} replays differ")

print("summary:", "PASS" if not failures else "FAIL")
for failure in failures:
    print(" ", failure)
raise SystemExit(1 if failures else 0)
