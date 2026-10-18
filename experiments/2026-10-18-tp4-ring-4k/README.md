# tp4-ring on the 4 KiB kernel: a qualification plan

**Status: proposed, not run.** Nothing here has touched the cluster. This is an optional
plan for the owner: running it needs the owner's go-ahead and a lab window. dgx4, like
dgx1-dgx3, is a production rank of `tp4-ring`.

## Why

An outside site ran `tp4-ring` at the recipe's sizing on a stock 4 KiB kernel without the
memory saver ([2026-10-17-site-tp4-4k-1017](../2026-10-17-site-tp4-4k-1017/README.md),
koldfrontier, PR #346). It kept 18-21 GiB of MemAvailable per node, including through a
993K-token prefill. Our ring at 64 KiB keeps about 20 GiB at its lowest
([2026-10-12-r6h](../2026-10-12-r6h/README.md)). So TP4 does not seem to need the memory
saver's ~1.8 GiB per node. TP3 needs it; TP4 inherited the 64 KiB host from the triangle.

A qualified 4 KiB TP4 shape would let a site serve 1M context with no kernel change, no
DKMS module and no MOK enrollment, so Secure Boot can stay on. The receipt cannot qualify
that shape here, for four reasons:
- it compares against SGLang, not against our ring at 64 KiB;
- it is one boot, on the site's own battery rather than our release suite;
- its memory guard only logged and never stopped the container;
- it ran an older DGX OS kernel (`6.17.0-1032-nvidia`) on driver 580.173.02.

## Hypothesis

At 4 KiB pages, `tp4-ring` at the recipe's sizing (1,048,576 context, 16 sequences,
KV 10.5 GiB) does the following on our fleet:
- boots cleanly;
- produces bit-identical outputs to the 64 KiB ring;
- keeps every node's lowest MemAvailable well above the guards, at both 1M prefill and
  16 streams.

The CPU page size touches host-side paths only: Engram's O_DIRECT row reads, pinned host
buffers, UVM and the page cache. No GPU kernel's arithmetic depends on it. Its speed
against 64 KiB is unmeasured; the release suite measures it.

## Intended delta

One variable: the host page-size profile that `tp3-triangle-4k` already declares.
- **Kernel.** The fleet's 4 KiB kernel `7.0.0-1019-nvidia`, the 4 KiB flavour of the
  production release. It is already installed and retained as the GRUB fallback on every
  node.
- **UVM and host policy.** Stock UVM loads, because the memory saver is built for the
  64 KiB kernel only. `spark3-kernel-memory` uses `/swap.img` and leaves THP alone.

Everything else is the recipe at `tp4-ring`, on the promoted image (r6v today,
`vllm-ds41f-kkref:19f2c20ed4d6-r6v`). That includes the sizing, the hub model profile,
the memory guards and the cabling. [candidate-arm.json](candidate-arm.json) sets only
`host.kernel`.

Out of scope: the site's other differences (like-port cabling, weights over NFS, a
snapshot+blobs profile, the log-only guard) and its older kernel. Each would be its own
experiment.

## Steps

All steps run inside one window, opened with `scripts/lab.py window open` after the
hold-file, `~/spark-request.json` and agent-traffic checks. The arm must be on a
published commit first. The plan uses one-shot `grub-reboot` and never changes GRUB's
default, so any reboot returns a node to 64 KiB.

1. **dgx4 first (host canary).** Stop the service, then boot dgx4 alone into the 4 KiB
   entry with `grub-reboot` (setup.md stage 2, step 4, with the 4 KiB entry).
   - Check `getconf PAGESIZE` (4096), `uname -r`, and that stock `nvidia_uvm` is loaded.
   - Check `swapon --show` (`/swap.img`), THP and `vm.min_free_kbytes` (4 KiB policy), and
     `scripts/host-recovery check`.
   - Check the fabric: GID index 3 is IPv4 on both functions, and 9000-byte pings reach
     both neighbours. Reboots have shifted neighbours' GID indices before.
   - Run CUDA validation in the image, which [memory-profiles.md](../../docs/memory-profiles.md)
     asks for before a distributed launch. Record idle MemAvailable with no container,
     to compare with 64 KiB.
   - Leave dgx4 on 4 KiB if it passes. If it fails, reboot dgx4, restore production, and
     record the failure.
2. **dgx1-dgx3.** Run the same one-shot boot and checks. Then run
   `bin/spark --cluster-config experiments/2026-10-18-tp4-ring-4k/candidate-arm.json doctor --live`.
   It must report no kernel or memory-saver findings and no `nodes differ` line.
3. **Boot and determinism.** Run `cluster start --apply` on the arm.
   - It must be ready with no NCCL warning and no guard trip. Record the ready time
     against r6v's 96 s.
   - Run the `gate.json` determinism gate. Temperature-0 and greedy outputs must equal
     r6v's references (3eab5112 / db25fde3, c9cf9dbe) exactly.
4. **Release comparison** ([docs/lab.md](../../docs/lab.md), "Release comparison").
   Run the ring's `release` suite plus `single-stream` in this order:
   - the 4 KiB arm first (heat soak);
   - then r6v at 64 KiB: reboot all four nodes and start the recipe at its shape. This
     arm is also the production restore;
   - optionally the 4 KiB arm again, to bound drift (two more reboots of every node).

   Compare with `scripts/release_table.py`. During each arm's 1M cold prefill and
   16-stream decode, record every node's lowest MemAvailable from the memory guard's log.
   This answers TODO "Head-node memory at 1M" at both page sizes.
5. **Restore.** Production is r6v at 64 KiB after step 4's second arm. Close the window.

The window takes roughly three hours: two reboots of every node, two suite arms of about
45 minutes each (r6v's took 46 and 47), and the gates.

## Quality and safety gates

- **Memory guards.** The guards stay at the recipe's values and in their normal mode,
  which stops the container: 5 GiB at startup, 3 GiB steady. A guard trip fails the arm.
- **Watchdog.** While any node runs 4 KiB, the window's watchdog cannot restore
  production, because `cluster start` refuses `tp4-ring`'s 64 KiB requirement. Keep the
  heartbeat fresh. Recovery from any failure is to reboot every node (the default is
  64 KiB), then the normal restore.
- **Mixed state.** Mixed kernels across ranks are invalid. If a node reboots mid-window,
  for example from the hardware watchdog, it returns to 64 KiB, and `doctor --live`
  shows `nodes differ`. Stop the arm before going on.
- **Failed requests.** Any failed request in the suite fails the arm.

## Acceptance criteria

The 4 KiB shape qualifies only if all of these hold:
- every host check passes;
- the boot is clean;
- the determinism gate is identical;
- the suite has no failed requests;
- every node's lowest MemAvailable stays at least 10 GiB through the 1M prefill and
  16 streams. The site saw about 18 GiB.

Speed is recorded as `release_table.py`'s faster / slower / same against 64 KiB, not
required. A slower 4 KiB ring could still be worth qualifying as the path without a
module, as `tp3-triangle-4k` is for TP3. The owner decides. Production stays on the
64 KiB ring either way.

If the owner accepts, one commit does all of the following:
- adds a `tp4-ring-4k` shape to `config/cluster.json` that extends `tp4-ring` with this
  `host.kernel` only, with its `benchmark_reference` from step 4's 4 KiB suite run;
- replaces the "externally reported" notes in `docs/setup.md` and
  `docs/memory-profiles.md` with a qualified row;
- adds that row to the README's table.
