# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug only: digests of chosen rows after every decoder layer.

With ``VLLM_DS41_ROW_TRACE_DIR`` set, every forward reads ``<dir>/spec.json``
(``{"rows": [[position, token_id], ...]}``, re-read when its mtime changes) and, for
each row of the step whose position and input token match an entry, appends one JSON
line per layer to ``<dir>/rank<R>.jsonl``: the forward's step number, the label
(``embed`` or ``layer<i>``), the position and token, the row's index in the step, the
step's row count, and SHA-256 prefixes of the row's hidden state and residual streams.
Comparing a prompt served alone with the same prompt in company shows the first layer
whose bits differ. Unset, nothing here runs. Each matched forward synchronizes the
device, so traced runs are slow; use them only to localize a difference.
"""

import hashlib
import itertools
import json
import os

import torch

DIR = os.environ.get("VLLM_DS41_ROW_TRACE_DIR")
ENABLED = bool(DIR)
_INSTANCES = itertools.count()
# The trace of the forward in progress, for sublayers (Engram) to record into.
ACTIVE: "RowTrace | None" = None


def _digest(row: torch.Tensor) -> str:
    data = row.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


class RowTrace:
    """One per process; ``begin`` per forward, ``match`` after any row remapping."""

    def __init__(self, rank: int):
        os.makedirs(DIR, exist_ok=True)
        os.chmod(DIR, 0o777)  # the host side writes spec.json and renames logs
        self.rank, self.step, self.mtime = rank, 0, None
        self.model = next(_INSTANCES)  # which model of the process recorded a line
        self.layer = -1  # the decoder layer in progress, for its sublayers' labels
        self.targets: set[tuple[int, int]] = set()
        self.matched: dict[int, tuple[int, int]] = {}
        self.rows = 0
        self.path = os.path.join(DIR, f"rank{rank}.jsonl")
        self.out = open(self.path, "a", buffering=1)

    def _reload(self) -> None:
        path = os.path.join(DIR, "spec.json")
        try:
            mtime = os.stat(path).st_mtime
        except FileNotFoundError:
            self.targets = set()
            return
        if mtime != self.mtime:
            self.mtime = mtime
            with open(path) as f:
                self.targets = {(int(p), int(t)) for p, t in json.load(f)["rows"]}

    def begin(self, positions: torch.Tensor, input_ids: torch.Tensor | None) -> None:
        global ACTIVE
        ACTIVE = self
        self.step += 1
        self._reload()
        if not os.path.exists(self.path):  # renamed away by the host: a new phase
            self.out = open(self.path, "a", buffering=1)
        self.match(positions, input_ids)

    def match(self, positions: torch.Tensor, input_ids: torch.Tensor | None) -> None:
        self.matched, self.rows = {}, int(positions.shape[0])
        if not self.targets or input_ids is None:
            return
        pairs = zip(positions.tolist(), input_ids.tolist())
        self.matched = {i: pair for i, pair in enumerate(pairs) if pair in self.targets}

    def record(
        self, label: str, tensors, start: int = 0, local_rows: int | None = None
    ):
        """Digests of the matched rows of ``tensors`` (rows ``start`` onward of the step
        held locally, as under sequence parallelism). A sequence-parallel layer's
        unreduced parts (``RankOrderParts``) are digested by their rank-order sum, the
        value a step without sequence parallelism holds there."""
        if not self.matched:
            return
        tensors = [
            t.sum() if t is not None and not isinstance(t, torch.Tensor) else t
            for t in tensors
        ]
        for row, (position, token) in self.matched.items():
            local = row - start
            if local < 0 or (local_rows is not None and local >= local_rows):
                continue
            digests = [None if t is None else _digest(t[local]) for t in tensors]
            self.out.write(
                json.dumps(
                    {
                        "model": self.model,
                        "step": self.step,
                        "label": label,
                        "pos": position,
                        "tok": token,
                        "row": row,
                        "rows": self.rows,
                        "digests": digests,
                    }
                )
                + "\n"
            )
