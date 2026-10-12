# r6h's remaining cost against r6d, per kernel, on the ring

r6h is level with or ahead of r6d on every TP4 metric, but on TP3 it trails r6d by 1.6%
±0.6 at eight prompts and by 1.2-1.5% on prefill
([r6h](../2026-10-12-r6h/README.md)). r6f's exact arithmetic (FP32 logits, the three-part
TF32 mHC projection, FP32 MoE partials) added cost that r6g and r6h won back on four
nodes, partly through the head grid fix, which never applies on three (at most 48 rows a
step).

The triangle cannot be cabled now. A TP3 step has at most 48 verify rows, and the
replicated kernels (mHC, norms, the drafter) cost the same per rank on either topology,
so a fixed-row profile on the ring at 48 rows shows most of what TP3 pays. The sharded
kernels (MoE, the heads) are bigger per rank on TP3; their TP3 shapes go to kernel
bundles on one node.

## Window 1 ([w1.json](w1.json))

A profile job, the candidate first: r6h ([r6h-fixed.json](r6h-fixed.json), the promoted
`config/cluster-tp4.json`) then r6d ([r6d-fixed.json](r6d-fixed.json), the fixed-row
profile's r6d arm), each with adaptive verification off (every step verifies the anchor
and all five drafts) and under the torch profiler:

- `code16`: sixteen streams, 96 rows a step;
- `c8`: eight streams, 48 rows a step (TP3's maximum);
- `prefill`: one 16K prefill (sequence parallelism).

The lab compares kernels per step between the arms with `analyze_costs.py`. Each
remaining loss becomes a bit-identical kernel change, measured at both topologies'
shapes, and then a TP4 screen; TP3 is confirmed at the next re-cable.
