#!/usr/bin/env python3
"""Launch TermoGrid AI Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "dashboard" / "app.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", "8501"],
        cwd=str(ROOT),
    )
