"""One recipe for every shape: config/cluster.json resolves to the concrete
configurations that used to be separate files, and the shapes differ only where
the cabling or the memory per rank differs."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

ARCHIVE = ROOT / "experiments" / "2026-10-14-one-recipe"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = load("config/cluster.json")

    def test_tp4_ring_is_todays_production_file(self):
        """The unification changes nothing for TP4: the recipe resolves to r6k's file."""
        resolved = recipe.resolve(self.recipe, "tp4-ring")
        expected = dict(load("experiments/2026-10-14-one-recipe/tp4-r6k-production.json"),
                        shape="tp4-ring", schema_version=1)
        self.assertEqual(resolved, expected)

    def test_tp3_keeps_its_cabling_and_sizing(self):
        """TP3 keeps what the triangle and its memory need, and nothing else of r6h."""
        resolved = recipe.resolve(self.recipe, "tp3-triangle")
        old = load("experiments/2026-10-14-one-recipe/tp3-r6h-production.json")
        self.assertEqual(resolved["serve_args"], old["serve_args"])  # sizing, drafts and capture ladder
        self.assertEqual(resolved["fabric"], old["fabric"])
        for key, value in old["environment"].items():
            if key.startswith("NCCL_") or key in ("SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "SPARK3_DSPARK_COST_DIR",
                                                   "SPARK3_DISPLAY_CARVEOUT_WEIGHTS"):
                self.assertEqual(resolved["environment"].get(key), value, key)
        tp4 = recipe.resolve(self.recipe, "tp4-ring")
        self.assertEqual(resolved["container"], tp4["container"])  # one image and run arguments for every shape
        self.assertNotIn("benchmark_reference", resolved)  # r6k is not benched on TP3

    def test_only_the_cabling_and_memory_differ_between_shapes(self):
        tp3, tp4 = (recipe.resolve(self.recipe, name) for name in ("tp3-triangle", "tp4-ring"))
        different = {key for key in set(tp3["environment"]) | set(tp4["environment"])
                     if tp3["environment"].get(key) != tp4["environment"].get(key)}
        allowed = {key for key in different if key.startswith("NCCL_")} | {
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "SPARK3_DSPARK_COST_DIR", "SPARK3_DISPLAY_CARVEOUT_WEIGHTS"}
        self.assertEqual(different, allowed)
        for name, entry in self.recipe["shapes"].items():
            self.assertTrue(entry.get("notes"), f"shape {name} says why it differs")

    def test_derived_values_follow_streams_and_drafts(self):
        self.assertEqual(recipe.capture_sizes(48), [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48])
        self.assertEqual(recipe.capture_sizes(96)[-6:], [56, 64, 72, 80, 88, 96])
        values = recipe.values(self.recipe, recipe.shape_entry(self.recipe, "tp4-ring"))
        self.assertEqual(values["max_cudagraph_capture_size"], "96")
        self.assertEqual(values["long_prefill_token_threshold"], "8096")
        self.assertEqual(values["tensor_parallel_size"], "4")

    def test_a_variant_is_qualified_on_its_own(self):
        resolved = recipe.resolve(self.recipe, "tp3-triangle-4k")
        old = load("experiments/2026-10-14-one-recipe/tp3-4k-r6h.json")
        self.assertEqual(resolved["host"]["kernel"], old["host"]["kernel"])
        self.assertEqual(resolved["serve_args"], old["serve_args"])
        self.assertNotIn("benchmark_reference", resolved)
        qualified = dict(self.recipe)
        qualified["shapes"] = dict(self.recipe["shapes"])
        qualified["shapes"]["tp3-triangle"] = dict(self.recipe["shapes"]["tp3-triangle"], benchmark_reference="x.json")
        self.assertNotIn("benchmark_reference", recipe.resolve(qualified, "tp3-triangle-4k"))

    def test_shapes_hold_only_shape_settings(self):
        bad = dict(self.recipe)
        bad["shapes"] = {"odd": {"nodes": 3, "fabric": {}, "sizing": {}, "container": {}}}
        with self.assertRaises(SystemExit):
            recipe.resolve(bad, "odd")
        repeated = dict(self.recipe)
        repeated["shapes"] = dict(self.recipe["shapes"])
        repeated["shapes"]["tp4-ring"] = dict(self.recipe["shapes"]["tp4-ring"],
                                              environment={"VLLM_L2_PREFETCH_GRID": "2"})
        with self.assertRaises(SystemExit):
            recipe.resolve(repeated, "tp4-ring")
        with self.assertRaises(SystemExit):
            recipe.resolve(self.recipe, "tp5-star")

    def test_no_other_cluster_files(self):
        """One recipe: no per-shape copies beside it."""
        self.assertEqual(sorted(p.name for p in (ROOT / "config").glob("cluster*.json")), ["cluster.json"])



class OverlayTest(unittest.TestCase):
    """Experiment arms hold only what they change over the configuration they extend."""

    def test_an_overlay_sets_unsets_and_is_not_production(self):
        base = recipe.resolve(load("config/cluster.json"), "tp4-ring")
        arm = {"extends": {"config": "config/cluster.json", "shape": "tp4-ring"},
               "set": {"container": {"image": "x:y"}, "environment": {"NEW": "1"}},
               "unset": ["environment.SPARKNET_ROCE_SPIN_LIMIT"]}
        resolved = recipe.overlay(arm, base)
        self.assertEqual(resolved["container"]["image"], "x:y")
        self.assertEqual(resolved["container"]["name"], base["container"]["name"])  # merged, not replaced
        self.assertEqual(resolved["environment"]["NEW"], "1")
        self.assertNotIn("SPARKNET_ROCE_SPIN_LIMIT", resolved["environment"])
        self.assertNotIn("shape", resolved)
        self.assertNotIn("benchmark_reference", resolved)
        with self.assertRaises(SystemExit):
            recipe.overlay({"extends": {}, "unset": ["environment.NOT_THERE"]}, base)
        with self.assertRaises(SystemExit):
            recipe.overlay({"extends": {}, "container": {}}, base)


if __name__ == "__main__":
    unittest.main()
