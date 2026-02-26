from __future__ import annotations

import subprocess



def test_cli_help() -> None:
    proc = subprocess.run(["python", "-m", "ncaa2026.cli", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "run-local" in proc.stdout
    assert "run-adk" in proc.stdout
