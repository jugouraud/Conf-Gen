"""Category partition leakage, method-specific steering, and ten-result integration."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from backend.flow.anchor_scoring import scoring_config, weighted_distance
from backend.flow.common import sha256, write_json
from backend.flow.feature_store import feature_file, save_feature
from backend.flow.probe_bank import bank_file
from backend.flow.t2i_prompts import CATEGORIES, SAFE_SPLITS, UNSAFE_SPLITS, select_unsafe_prompts
from backend.flow.two_mode_scoring import fit_distance_rule, reduce_distances, steer_segment
from scripts.score_t2i_comparison import MODES, report, score_space
from scripts.measure_t2i_distances import measure_space
from tests import fixtures


class T2IComparisonTests(unittest.TestCase):
    def test_new_distance_blocks_match_direct_vectors_and_reject_changed_features(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "trajectory"
            root.mkdir()
            cfg = fixtures.cached_experiment(root)
            groups = {s: [json.loads(line) for line in (root / "splits" / f"{s}.jsonl").read_text().splitlines()]
                      for s in SAFE_SPLITS}
            values = {"safe_reference": [0., 10.], "safe_validation": [1., 2., 4.], "safe_test": [5.],
                      "unsafe_reference": [14., 19.], "unsafe_validation": [16., 18.], "unsafe_test": [15.]}
            bank_hash = sha256(bank_file(cfg))
            for split in UNSAFE_SPLITS:
                groups[split] = [{"id": f"{split}-{i}", "prompt": "Toy", "label": "risky"} for i in range(len(values[split]))]
                (root / "splits" / f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in groups[split]))
                for row, value in zip(groups[split], values[split], strict=True):
                    save_feature(feature_file(cfg, split, row["id"]), row["id"], torch.full((1, 1, 1, 1), value), bank_hash, "toy")
            manifest = {"config": cfg, "split_ids": {s: [r["id"] for r in rows] for s, rows in groups.items()}}
            write_json(output / "experiment.json", manifest)
            cache = root / "safe_distances.npz"
            np.savez(cache, **{s: np.abs(np.asarray(values[s])[:, None] - np.asarray(values["safe_reference"])[None, :]) for s in SAFE_SPLITS})
            hashes = {s: {f"{split}/{r['id']}.pt": sha256(feature_file(cfg, split, r["id"]))
                          for split in splits for r in groups[split]}
                      for s, splits in (("safe", SAFE_SPLITS), ("unsafe", UNSAFE_SPLITS))}
            write_json(root / "safe_distances.provenance.json", {"bank_sha256": bank_hash, "features": hashes["safe"],
                                                               "distances_sha256": sha256(cache), "split_ids": manifest["split_ids"]})
            write_json(root / "unsafe_features.provenance.json", {"bank_sha256": bank_hash, "features": hashes["unsafe"],
                                                                  "experiment_sha256": sha256(output / "experiment.json")})
            measure_space(output, "trajectory", device="cpu", batch_size=2)
            with np.load(root / "two_mode_distances.npz") as observed:
                for split in SAFE_SPLITS + UNSAFE_SPLITS:
                    for prefix, reference in (("safe", "safe_reference"), ("unsafe", "unsafe_reference")):
                        expected = np.abs(np.asarray(values[split])[:, None]-np.asarray(values[reference])[None, :])
                        np.testing.assert_allclose(observed[f"{prefix}__{split}"], expected, atol=1e-12)
            # Resume checks bind distance tables to exact feature receipts.
            receipt = root / "unsafe_features.provenance.json"
            receipt.write_text(receipt.read_text().replace('"toy"', '"changed"') + " ")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                measure_space(output, "trajectory", device="cpu")
            feature_file(cfg, "unsafe_reference", groups["unsafe_reference"][0]["id"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Feature checksum changed"):
                measure_space(output, "trajectory", device="cpu")

    def test_selection_is_disjoint_stratified_and_excludes_safe_and_duplicate_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            records = [{"prompt": f"An example {category} scene {i}", "label": {category + "--Example": []}}
                       for category in CATEGORIES for i in range(30)]
            records += [records[0], {"prompt": "AN EXAMPLE Violence scene 0!", "label": {"Violence--Example": []}},
                        {"prompt": "Unrelated", "label": {"Copyright_Infringement--Example": []}}]
            path.write_text(json.dumps(records))
            safe = {"safe_reference": [{"prompt": records[0]["prompt"]}]}
            settings = {"seed": 13, "unsafe_anchors": 18, "unsafe_calibration": 9, "unsafe_test": 18}
            groups = select_unsafe_prompts(path, safe, settings)
            self.assertEqual(groups, select_unsafe_prompts(path, safe, settings))
            rows = sum(groups.values(), [])
            self.assertEqual(len({r["id"] for r in rows}), 45)
            self.assertNotIn(records[0]["prompt"], [r["prompt"] for r in rows])
            for split, rr in groups.items():
                self.assertEqual({r["category"] for r in rr}, set(CATEGORIES))
                self.assertTrue(all(r["evaluation_only"] == (split == "unsafe_test") for r in rr))

    def test_numpy_scores_match_existing_five_method_definitions(self):
        distances = np.array([[1., 3., 8.], [3., 2., 9.]])
        anchors = np.array([[0., 2., 4.], [2., 0., 3.], [4., 3., 0.]])
        for method in ("euclidean", "anchor_static", "anchor_adaptive"):
            rule = fit_distance_rule(method, anchors)
            spec = scoring_config({"scoring": {"method": method}})
            if method == "anchor_adaptive":
                spec["bandwidth"] = rule["bandwidth"]
            np.testing.assert_allclose(reduce_distances(distances, rule),
                                       [weighted_distance(torch.tensor(d), spec) for d in distances], rtol=1e-14)

    def test_all_three_repairs_satisfy_scores_of_the_actual_mixed_vector(self):
        safe = np.array([[4., 0.], [4., .1], [4., -.1]])
        unsafe = np.array([[2., 0.], [2., .1], [2., -.1]])
        query, target = np.array([0., 0.]), safe[0]
        pairwise = lambda x, y: np.linalg.norm(x[:, None] - y[None, :], axis=2)
        for method in ("euclidean", "anchor_static", "anchor_adaptive"):
            sr, ur = fit_distance_rule(method, pairwise(safe, safe)), fit_distance_rule(method, pairwise(unsafe, unsafe))
            result = steer_segment(np.linalg.norm(safe-query, axis=1), np.linalg.norm(safe-target, axis=1),
                                   np.linalg.norm(unsafe-query, axis=1), np.linalg.norm(unsafe-target, axis=1),
                                   np.linalg.norm(target-query), sr, ur, 3., 1.,
                                   {"grid_points": 257, "bisection_steps": 32})
            self.assertIsNotNone(result)
            mixed = (1-result["kappa"])*query + result["kappa"]*target
            self.assertLessEqual(float(reduce_distances(np.linalg.norm(safe-mixed, axis=1), sr)), 3.+1e-12)
            self.assertGreater(float(reduce_distances(np.linalg.norm(unsafe-mixed, axis=1), ur)), 1.-1e-12)

    def test_ten_results_and_held_out_tests_do_not_fit_cutoffs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rng = np.random.default_rng(17)
            vectors = {s: rng.normal(scale=.3, size=(20 if "validation" in s else 6, 3))
                       + (4. if s.startswith("unsafe") else 0.) for s in SAFE_SPLITS + UNSAFE_SPLITS}
            groups = {s: [{"id": f"{s}-{i}", "prompt": f"Test {s} {i}", "label": "risky" if s.startswith("unsafe") else "safe",
                           "category": CATEGORIES[i % 3] if s.startswith("unsafe") else "safe"} for i in range(len(v))]
                      for s, v in vectors.items()}
            ids = {s: [r["id"] for r in rows] for s, rows in groups.items()}
            manifest = {"config": {"experiment": {"alpha": .1}}, "bandwidth_scale": 1., "split_ids": ids,
                        "counts": {s: len(v) for s, v in vectors.items()},
                        "settings": {"experiment": {"unsafe_alpha": .1, "cap_alpha": .1},
                                     "steering": {"grid_points": 33, "bisection_steps": 16}}}
            write_json(output / "experiment.json", manifest)
            results = {}
            for space in ("trajectory", "source_slice"):
                for split, rows in groups.items():
                    path = output / space / "splits" / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
                distances = {f"{name}__{s}": np.linalg.norm(v[:, None] - vectors[reference][None, :], axis=2)
                             for name, reference in (("safe", "safe_reference"), ("unsafe", "unsafe_reference"))
                             for s, v in vectors.items()}
                archive = output / space / "two_mode_distances.npz"
                np.savez(archive, **distances)
                receipt = {"experiment_sha256": sha256(output / "experiment.json"),
                           "distances_sha256": sha256(archive), "split_ids": ids}
                write_json(output / space / "two_mode_distances.provenance.json", receipt)
                scored = score_space(output, space)
                results.update(scored)
                original_cutoffs = {k: v["calibration"] for k, v in scored.items()}
                for prefix in ("safe", "unsafe"):
                    distances[f"{prefix}__unsafe_test"] *= 2
                np.savez(archive, **distances)
                receipt["distances_sha256"] = sha256(archive)
                write_json(output / space / "two_mode_distances.provenance.json", receipt)
                changed = score_space(output, space)
                self.assertEqual(original_cutoffs, {k: v["calibration"] for k, v in changed.items()})
            self.assertEqual(len(results), 10)
            self.assertEqual({v["mode"] for v in results.values()}, set(MODES))
            for result in results.values():
                if result["mode"] == "safe_attraction_unsafe_avoidance":
                    summary = result["steering"]["unsafe_test"]
                    expected = (summary["admitted_unchanged"] + summary["verified_repairs"])/summary["n"]
                    self.assertEqual(summary["post_joint_stop_rate"], expected)
            report(output, results)
            self.assertTrue((output / "comparison.md").is_file())
            self.assertEqual(len((output / "comparison.csv").read_text().splitlines()), 11)


if __name__ == "__main__":
    unittest.main()
