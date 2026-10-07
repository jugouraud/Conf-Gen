"""Keep the analysis prompt table limited to the visible page."""

import unittest

from nicegui import ui
from nicegui.elements.plotly import Plotly
from nicegui.elements.table import Table

from backend.validation.region_report import _metric_report
from frontend.coco_region_analysis_page import create_coco_region_analysis_page


class AnalysisPageTests(unittest.TestCase):
    def test_initial_page_sends_only_nine_prompt_rows(self) -> None:
        rows = [
            {"row_number": number, "prompt": f"prompt {number}", "category": "test",
             "prompt_toxicity": number / 100, "distance": number / 10,
             "radius": 1.0, "margin": number / 10 - 1,
             "blocked": number > 10}
            for number in range(1, 21)
        ]
        report = {"rows": rows, "metrics": {"nearest_anchor": _metric_report(
            rows, rows, score_key="distance", budget_key="radius", blocked_key="blocked",
        )}}

        create_coco_region_analysis_page(lambda: self.fail("Report recomputed"), initial_report=report)

        prompt_tables = [element for element in ui.context.client.elements.values()
                         if isinstance(element, Table) and any(
                             column["name"] == "prompt" for column in element.columns
                         )]
        self.assertEqual(len(prompt_tables[-1].rows), 9)
        self.assertEqual(prompt_tables[-1].rows[0]["row_number"], 11)

        plots = [element for element in ui.context.client.elements.values()
                 if isinstance(element, Plotly)]
        distribution = plots[-1].figure
        self.assertEqual(distribution["layout"]["xaxis"]["title"], "Prompt toxicity")
        self.assertEqual(distribution["layout"]["yaxis"]["title"], "Density")
        self.assertEqual({trace["name"] for trace in distribution["data"]}, {"Blocked", "Allowed"})
        self.assertTrue(all(trace["type"] == "scatter" for trace in distribution["data"]))
        self.assertTrue(all(trace["x"][0] == 0 and trace["x"][-1] == 1
                            for trace in distribution["data"]))
        peaks = {trace["name"]: trace["x"][trace["y"].index(max(trace["y"]))]
                 for trace in distribution["data"]}
        self.assertGreater(peaks["Blocked"], peaks["Allowed"])


if __name__ == "__main__":
    unittest.main()
