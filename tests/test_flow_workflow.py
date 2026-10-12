"""Verify the full experiment runs in the required dependency order."""

import unittest
from unittest.mock import patch

from backend.flow.workflow import run_complete_workflow


class WorkflowTests(unittest.TestCase):
    def test_full_workflow_order(self):
        calls = []

        def record(name):
            return lambda *args: calls.append((name, args[1:]))

        steps = ("prepare", "freeze_probes", "extract", "calibrate", "evaluate", "report")
        patches = [patch(f"backend.flow.workflow.{name}", side_effect=record(name))
                   for name in steps]
        for mocked_step in patches:
            mocked_step.start()
        try:
            run_complete_workflow({})
        finally:
            for mocked_step in reversed(patches):
                mocked_step.stop()

        self.assertEqual(calls, [
            ("prepare", ()), ("freeze_probes", ()),
            ("extract", ("safe_reference",)),
            ("extract", ("safe_validation",)),
            ("calibrate", ()),
            ("extract", ("safe_test",)),
            ("evaluate", ("safe_test",)),
            ("extract", ("unsafe_test",)),
            ("evaluate", ("unsafe_test",)),
            ("report", ()),
        ])


if __name__ == "__main__":
    unittest.main()
