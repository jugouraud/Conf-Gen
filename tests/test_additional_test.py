"""Frozen calibration, unchanged unsafe decisions, and separate additional-test reports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.flow.common import sha256, write_json
from backend.flow.two_mode_scoring import fit_distance_rule, reduce_distances
from backend.flow.comparison import VARIANTS
from scripts.evaluate_additional_test import SPLIT, frozen_inputs, report, score
from scripts.score_t2i_comparison import MODES


def fixture(output):
    safe = np.array([[0., 0.], [.2, 0.], [0., .2]])
    unsafe = np.array([[3., 3.], [3.2, 3.], [3., 3.2]])
    queries = np.array([[.1, .1], [.3, .2], [1.3, 1.3], [2.9, 2.9]])
    original = np.array([[3.1, 3.1], [4., 3.], [3., 4.]])
    distance = lambda a, b: np.linalg.norm(a[:, None] - b[None, :], axis=2)
    frozen, parent, geometry_hashes = {}, {}, {}
    ids = [f"query-{i}" for i in range(len(queries))]
    for space, methods in VARIANTS.items():
        folder = output / space
        for split, vectors in (("safe_reference", safe), ("unsafe_test", original), (SPLIT, queries)):
            rows = [{"id": ids[i] if split == SPLIT else f"{split}-{i}", "category": ("Violence", "Pornography", "Illegal_Activities")[i % 3]}
                    for i in range(len(vectors))]
            path = folder / "splits" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        geometry = {"safe__safe_reference": distance(safe, safe), "unsafe__safe_reference": distance(safe, unsafe),
                    "safe__unsafe_test": distance(original, safe), "unsafe__unsafe_test": distance(original, unsafe)}
        np.savez(folder / "frozen_geometry.npz", **geometry)
        geometry_hashes[space] = sha256(folder / "frozen_geometry.npz")
        np.savez(folder / "additional_distances.npz", safe=distance(queries, safe), unsafe=distance(queries, unsafe))
        for method in methods:
            name = f"{space}_{method}"
            calibration = {"safe_rule": fit_distance_rule(method, distance(safe, safe)),
                           "unsafe_rule": fit_distance_rule(method, distance(unsafe, unsafe)),
                           "safe_cutoff": 2., "unsafe_cutoff": 3.,
                           "safe_calibration_ids": ["old-safe-calibration"], "unsafe_calibration_ids": ["old-unsafe-calibration"]}
            frozen[name] = calibration
            for mode in MODES:
                s = reduce_distances(geometry["safe__unsafe_test"], calibration["safe_rule"])
                u = reduce_distances(geometry["unsafe__unsafe_test"], calibration["unsafe_rule"])
                flags = (s > 2.) | ((u <= 3.) if mode != "safe_attraction" else False)
                parent[name + "__" + mode] = {"calibration": calibration, "safe_false_positive_rate": .05,
                                               "unsafe_flagged_rate": float(flags.mean())}
    write_json(output / "parent_comparison.json", {"methods": parent})
    write_json(output / "frozen_calibration.json", frozen)
    manifest = {"counts": {SPLIT: len(queries)}, "split_ids": {SPLIT: ids}, "input_file": "new-controls.json",
                "label_meaning": "Synthetic intended-benign controls",
                "settings": {"steering": {"grid_points": 33, "bisection_steps": 16}},
                "parent_comparison_sha256": sha256(output / "parent_comparison.json"),
                "frozen_geometry_sha256": geometry_hashes}
    write_json(output / "experiment.json", manifest)
    for space in VARIANTS:
        write_json(output / space / "additional_distances.provenance.json",
                   {"experiment_sha256": sha256(output / "experiment.json"),
                    "distances_sha256": sha256(output / space / "additional_distances.npz")})
    return frozen


class AdditionalTestTests(unittest.TestCase):
    def test_interrupted_run_reuses_transferred_hashes_without_repeating_inference(self):
        from scripts.extract_t2i_comparison import run_extraction
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_json(output / "experiment.json", {"counts": {SPLIT: 2}})
            configs, cached = {}, {}
            for space in VARIANTS:
                folder = output / space
                configs[space] = {"paths": {"output": str(folder)}}
                (folder / "splits").mkdir(parents=True)
                (folder / "splits" / f"{SPLIT}.jsonl").write_text(
                    ''.join(json.dumps({"id": key, "prompt": key})+'\n' for key in ("cached", "fresh")))
                (folder / "probes").mkdir()
                (folder / "probes/probe_bank.pt").write_bytes(b"frozen-bank")
                cached[f"{space}/features/{SPLIT}/cached.pt"] = "f"*64
            adapter = Mock(signature="toy")
            with patch("scripts.extract_t2i_comparison.space_config", side_effect=lambda _, s: configs[s]), \
                    patch("scripts.extract_t2i_comparison.read_bank", return_value={}), \
                    patch("scripts.extract_t2i_comparison.evaluate_probe_field", return_value=torch.ones(1)) as field:
                run_extraction(output, adapter, splits=(SPLIT,), provenance_name="additional_features.provenance.json",
                               precomputed={"features": cached, "experiment_sha256": sha256(output / "experiment.json")})
            adapter.encode.assert_called_once_with("fresh")
            self.assertEqual(field.call_count, 2)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual((status["completed_features"], status["status"]), (4, "complete"))
            for space in VARIANTS:
                receipt = json.loads((output / space / "additional_features.provenance.json").read_text())
                self.assertEqual(receipt["features"][f"{SPLIT}/cached.pt"], "f"*64)
                self.assertIn(f"{SPLIT}/fresh.pt", receipt["features"])

    def test_new_controls_produce_ten_rows_without_refitting_or_changing_unsafe_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = fixture(output)
            results = {}
            for space in VARIANTS:
                results.update(score(output, space))
            self.assertEqual(len(results), 10)
            for result in results.values():
                self.assertEqual(result["calibration"], frozen[f"{result['space']}_{result['method']}"])
                self.assertEqual((result["additional_intended_benign_n"], result["unsafe_test_n"]), (4, 3))
                self.assertEqual(result["unsafe_flagged_rate"], 1.)
            self.assertGreater(results["trajectory_euclidean__safe_attraction_unsafe_avoidance"]["intended_benign_flagged_rate"],
                               results["trajectory_euclidean__safe_attraction"]["intended_benign_flagged_rate"])
            report(output, results)
            self.assertEqual(len((output / "comparison.csv").read_text().splitlines()), 11)
            self.assertIn("without human safety", (output / "comparison.md").read_text())

    def test_changed_additional_distribution_cannot_change_frozen_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = fixture(output)
            score(output, "trajectory")
            folder = output / "trajectory"
            with np.load(folder / "additional_distances.npz") as data:
                safe, unsafe = data["safe"].copy()*2, data["unsafe"].copy()*.5
            np.savez(folder / "additional_distances.npz", safe=safe, unsafe=unsafe)
            receipt = json.loads((folder / "additional_distances.provenance.json").read_text())
            receipt["distances_sha256"] = sha256(folder / "additional_distances.npz")
            write_json(folder / "additional_distances.provenance.json", receipt)
            for row in score(output, "trajectory").values():
                self.assertEqual(row["calibration"], frozen[f"trajectory_{row['method']}"])
                self.assertEqual(row["unsafe_flagged_rate"], 1.)

    def test_changed_cutoff_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = fixture(output)
            frozen["trajectory_euclidean"]["safe_cutoff"] = 200.
            write_json(output / "frozen_calibration.json", frozen)
            with self.assertRaisesRegex(ValueError, "change a frozen calibration"):
                frozen_inputs(output)


if __name__ == "__main__":
    unittest.main()
