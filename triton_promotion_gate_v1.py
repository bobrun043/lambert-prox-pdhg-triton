"""Promotion gate: refuse Triton integration unless CUDA oracle validation passed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "triton-oracle-validation-v2"
MIN_EXPECTED_TESTS = 47


class TritonPromotionError(RuntimeError):
    pass


def load_and_validate(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        raise TritonPromotionError(f"validation report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != EXPECTED_SCHEMA:
        raise TritonPromotionError(f"unexpected validation schema: {report.get('schema')!r}")
    if report.get("status") != "PASS":
        raise TritonPromotionError(f"Triton validation status is {report.get('status')!r}, not PASS")
    tests = report.get("tests")
    if not isinstance(tests, list) or len(tests) < MIN_EXPECTED_TESTS:
        raise TritonPromotionError(
            f"incomplete Triton validation: {len(tests) if isinstance(tests, list) else 'invalid'} tests"
        )
    failed = [t for t in tests if not bool(t.get("pass"))]
    if failed:
        raise TritonPromotionError(f"validation contains {len(failed)} failed tests")
    env = report.get("environment", {})
    if not env.get("cuda_available") or not env.get("triton_importable"):
        raise TritonPromotionError("PASS report lacks CUDA/Triton environment evidence")
    return report
