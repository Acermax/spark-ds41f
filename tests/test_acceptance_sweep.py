"""Host-independent tests for the acceptance sweep's counter parsing and summary."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("acceptance_sweep", str(ROOT / "scripts/acceptance_sweep.py"))
spec = importlib.util.spec_from_loader("acceptance_sweep", loader)
sweep = importlib.util.module_from_spec(spec)
loader.exec_module(sweep)

METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 40.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 25.0
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="m"} 5.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 9.0
vllm:spec_decode_num_drafts_created 1.7e9
"""


class AcceptanceSweepTest(unittest.TestCase):
    def test_counters_sum_engines_and_skip_other_series(self) -> None:
        self.assertEqual(sweep.counters(METRICS), {
            "vllm:spec_decode_num_drafts_total": 10.0,
            "vllm:spec_decode_num_draft_tokens_total": 40.0,
            "vllm:spec_decode_num_accepted_tokens_total": 30.0,
        })

    def test_summary_means_over_prompts(self) -> None:
        rows = [{"accepted_per_draft": 1.0, "tps": 80.0}, {"accepted_per_draft": 2.0, "tps": 100.0}]
        result = sweep.summary(rows)
        self.assertEqual(result["summary"], "acceptance sweep")
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["accepted_per_draft_mean"], 1.5)
        self.assertEqual(result["tps_mean"], 90.0)
        self.assertGreater(result["accepted_ci95_pct"], 0)

    def test_uses_the_distinct_stream_prompts(self) -> None:
        self.assertEqual(len(sweep.PROMPTS), 16)


if __name__ == "__main__":
    unittest.main()
