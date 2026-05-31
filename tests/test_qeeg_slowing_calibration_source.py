from __future__ import annotations

import subprocess
import sys


def test_residual_calibration_script_exposes_qeeg_slowing_source():
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/26_calibrate_eeg_summary_residual.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "qeeg_slowing" in completed.stdout
