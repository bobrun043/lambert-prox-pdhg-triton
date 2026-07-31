from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mpmath
import numpy
import scipy


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reference_validation_report.json"


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q", "test_lambert_prox_reference_v1.py"]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    payload = {
        "schema": "lambert-prox-reference-validation-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "mpmath": mpmath.__version__,
        },
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    print(f"Validation report: {REPORT}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
