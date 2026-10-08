"""Checks for the saved-data T2I safe-region analysis page."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui
from nicegui.elements.plotly import Plotly
from nicegui.elements.select import Select
from nicegui.elements.table import Table

from backend.embeddings.prompt_region import _source_sha256
from backend.validation.safe_prompt_analysis import load_safe_prompt_analysis
from frontend.safe_prompt_analysis_page import (
    _distribution_figure,
    _harmfulness_box_figure,
    create_safe_prompt_analysis_page,
)


class SafePromptAnalysisTests(unittest.TestCase):
    def test_loader_uses_transport_scores_and_checks_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.parquet"
            region_path = root / "region.npz"
            transport_path = root / "transport.sqlite3"
            report_path = root / "report.json"
            method_report_path = root / "method_report.json"
            rows = [
                ("reference", "safe_reference", "safe", True, False, None),
                ("calibration", "safe_calibration", "safe", True, False, None),
                ("safe_test", "safe_test", "safe", False, True, None),
                ("i2p", "toxic_test_i2p", "risky", False, True, "violence"),
                ("t2i", "toxic_test_t2i_risky", "risky", False, True,
                 json.dumps({"Violence--Bloody_Content": []})),
            ]
            pd.DataFrame([
                {"prompt_key": key, "prompt": key.replace("_", " "), "split": split,
                 "label": label, "eligible_for_region": eligible,
                 "evaluation_only": evaluation, "category": category,
                 "i2p_prompt_toxicity": 0.25 if key == "i2p" else None,
                 "risk_score": 40.0 if key == "i2p" else None}
                for key, split, label, eligible, evaluation, category in rows
            ]).to_parquet(corpus_path)
            dimension = 768
            np.savez_compressed(
                region_path, anchors=np.zeros((1, dimension), dtype=np.float32),
                mean=np.zeros(dimension, dtype=np.float32),
                components=np.zeros((dimension, 1), dtype=np.float32),
                eigenvalues=np.ones(1, dtype=np.float32), gamma=1.0,
                radius=0.5, metadata=json.dumps({
                    "corpus_sha256": _source_sha256(corpus_path), "model_id": "test",
                }),
            )
            report = {
                "source": str(corpus_path.resolve()), "region": str(region_path.resolve()),
                "radius": 0.5,
                "splits": {
                    "safe_test": {"count": 1, "accepted": 1, "rejected": 0, "acceptance_rate": 1.0},
                    "toxic_test_i2p": {"count": 1, "accepted": 0, "rejected": 1, "acceptance_rate": 0.0},
                    "toxic_test_t2i_risky": {"count": 1, "accepted": 0, "rejected": 1, "acceptance_rate": 0.0},
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            inputs = {"corpus_sha256": _source_sha256(corpus_path),
                      "region_sha256": _source_sha256(region_path)}
            method_report = {
                "inputs": inputs, "transport": str(transport_path.resolve()),
                "methods": {
                    method: {"radius": radius, "splits": {
                        "safe_test": {"count": 1, "rejected": 0},
                        "toxic_test_i2p": {"count": 1, "rejected": i2p_rejected},
                        "toxic_test_t2i_risky": {"count": 1, "rejected": 1},
                    }}
                    for method, radius, i2p_rejected in (
                        ("nearest_anchor", 0.5, 1), ("order1", 0.15, 0),
                        ("orderinf", 0.5, 1),
                    )
                },
            }
            method_report_path.write_text(json.dumps(method_report), encoding="utf-8")
            with closing(sqlite3.connect(transport_path)) as connection:
                connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
                connection.execute("INSERT INTO metadata VALUES ('inputs', ?)", (json.dumps(inputs),))
                connection.execute(
                    "CREATE TABLE scores (prompt_key TEXT, split TEXT, nearest REAL, order1 REAL, orderinf REAL)"
                )
                connection.executemany("INSERT INTO scores VALUES (?,?,?,?,?)", [
                    ("safe_test", "safe_test", 0.1, 0.01, 0.1),
                    ("i2p", "toxic_test_i2p", 1.0, 0.1, 1.0),
                    ("t2i", "toxic_test_t2i_risky", 2.0, 0.2, 2.0),
                ])
                connection.commit()
            analysis = load_safe_prompt_analysis(
                corpus_path=corpus_path, region_path=region_path,
                report_path=report_path, transport_path=transport_path,
                method_report_path=method_report_path,
            )
            self.assertEqual([row["rejected"] for row in analysis["rows"]], [False, True, True])
            self.assertFalse(analysis["rows"][1]["rejections"]["order1"])
            self.assertEqual(analysis["rows"][-1]["category"], "Violence--Bloody_Content")
            self.assertEqual(analysis["rows"][1]["prompt_toxicity"], 0.25)
            self.assertEqual(analysis["rows"][1]["inappropriate_percentage"], 40.0)
            self.assertIsNone(analysis["rows"][2]["prompt_toxicity"])
            order1_boxes = _harmfulness_box_figure(analysis["rows"], "prompt_toxicity", "order1")
            nearest_boxes = _harmfulness_box_figure(analysis["rows"], "prompt_toxicity", "nearest_anchor")
            self.assertEqual([trace["y"] for trace in order1_boxes["data"]], [[0.25], []])
            self.assertEqual([trace["y"] for trace in nearest_boxes["data"]], [[], [0.25]])
            report["splits"]["toxic_test_i2p"]["rejected"] = 0
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_safe_prompt_analysis(
                    corpus_path=corpus_path, region_path=region_path,
                    report_path=report_path, transport_path=transport_path,
                    method_report_path=method_report_path,
                )

    def test_page_paginates_prompt_decisions(self) -> None:
        split_metrics = {
            "safe_test": {"count": 12, "accepted": 11, "rejected": 1,
                          "acceptance_rate": 11 / 12},
            "toxic_test_i2p": {"count": 12, "accepted": 9, "rejected": 3,
                                "acceptance_rate": 0.75},
            "toxic_test_t2i_risky": {"count": 12, "accepted": 8, "rejected": 4,
                                      "acceptance_rate": 2 / 3},
        }
        analysis = {
            "report": {"radius": 1.0, "splits": split_metrics},
            "methods": {method: {"radius": 1.0, "splits": split_metrics}
                        for method in ("nearest_anchor", "order1", "orderinf")},
            "safe_count": 80, "risky_count": 24,
            "reference_count": 60, "calibration_count": 8,
            "distributions": {method: {key: [0.5, 1.5] for key in split_metrics}
                              for method in ("nearest_anchor", "order1", "orderinf")},
            "rows": [
                {"row_number": index, "prompt": f"prompt {index}", "split": "safe_test",
                 "source": "Held-out safe", "category": "Safe prompt", "score": 0.5,
                 "rejected": False,
                 "scores": {"nearest_anchor": 0.5, "order1": 1.5, "orderinf": 0.5},
                 "rejections": {"nearest_anchor": False, "order1": True, "orderinf": False}}
                for index in range(1, 26)
            ],
        }
        create_safe_prompt_analysis_page(analysis)
        tables = [element for element in ui.context.client.elements.values()
                  if isinstance(element, Table)]
        self.assertEqual(len(tables[-3].rows), 3)
        self.assertEqual(tables[-3].rows[1]["method"], "Order-1 W1")
        self.assertEqual(len(tables[-1].rows), 10)
        self.assertEqual(tables[-1].rows[0]["row_number"], 1)
        plots = [element for element in ui.context.client.elements.values()
                 if isinstance(element, Plotly)]
        self.assertEqual(len(plots[-2].figure["data"]), 3)
        self.assertEqual([trace["type"] for trace in plots[-1].figure["data"]], ["box", "box"])
        method_select = next(element for element in reversed(list(ui.context.client.elements.values()))
                             if isinstance(element, Select) and element._props.get("label") == "Decision method")
        method_select.set_value("order1")
        self.assertEqual(tables[-1].rows[0]["score"], "1.5000")
        self.assertEqual(tables[-1].rows[0]["decision"], "REJECT")
        self.assertEqual(plots[-2].figure["layout"]["xaxis"]["title"], "Order-1 W1 score")

    def test_i2p_box_plot_separates_decisions_and_scales(self) -> None:
        rows = [
            {"split": "toxic_test_i2p", "rejected": False,
             "prompt_toxicity": 0.1, "inappropriate_percentage": 10.0},
            {"split": "toxic_test_i2p", "rejected": True,
             "prompt_toxicity": 0.7, "inappropriate_percentage": 80.0},
            {"split": "safe_test", "rejected": True,
             "prompt_toxicity": None, "inappropriate_percentage": None},
        ]
        toxicity = _harmfulness_box_figure(rows, "prompt_toxicity")
        self.assertEqual([trace["y"] for trace in toxicity["data"]], [[0.1], [0.7]])
        self.assertEqual(toxicity["layout"]["yaxis"]["range"], [0, 1])
        inappropriate = _harmfulness_box_figure(rows, "inappropriate_percentage")
        self.assertEqual([trace["y"] for trace in inappropriate["data"]], [[10.0], [80.0]])
        self.assertEqual(inappropriate["layout"]["yaxis"]["range"], [0, 100])

    def test_order1_distribution_uses_its_score_scale(self) -> None:
        analysis = {
            "distributions": {"order1": {
                "safe_test": [0.01, 0.02],
                "toxic_test_i2p": [0.015],
                "toxic_test_t2i_risky": [0.03],
            }},
            "methods": {"order1": {"radius": 0.025}},
        }
        figure = _distribution_figure(analysis, "order1")
        centers = figure["data"][0]["x"]
        self.assertLess(centers[-1] - centers[0], 0.1)
        self.assertEqual(figure["layout"]["shapes"][0]["x0"], 0.025)


if __name__ == "__main__":
    unittest.main()
