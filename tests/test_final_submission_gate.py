from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "67_run_final_submission_gate.py"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("final_submission_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FinalSubmissionGateTests(unittest.TestCase):
    def test_overall_status_blocks_on_author_metadata(self) -> None:
        gate = load_gate_module()
        checks = [
            gate.GateCheck("manuscript", "integrity", "PASS", "76 PASS", "all pass"),
            gate.GateCheck("package", "zip", "PASS", "testzip=None", "valid archive"),
            gate.GateCheck("author_metadata", "strict_validation", "BLOCKED", "exit=1", "missing author fields"),
        ]

        self.assertEqual(gate.determine_overall_status(checks), "BLOCKED_BY_AUTHOR_METADATA")

    def test_overall_status_fails_on_non_author_failures(self) -> None:
        gate = load_gate_module()
        checks = [
            gate.GateCheck("manuscript", "integrity", "PASS", "76 PASS", "all pass"),
            gate.GateCheck("package", "zip", "FAIL", "testzip=bad.docx", "corrupt archive"),
        ]

        self.assertEqual(gate.determine_overall_status(checks), "FAIL")

    def test_count_statuses(self) -> None:
        gate = load_gate_module()
        checks = [
            gate.GateCheck("a", "one", "PASS", "", ""),
            gate.GateCheck("b", "two", "WARN", "", ""),
            gate.GateCheck("c", "three", "PASS", "", ""),
        ]

        self.assertEqual(gate.count_statuses(checks), {"PASS": 2, "WARN": 1})


if __name__ == "__main__":
    unittest.main()
