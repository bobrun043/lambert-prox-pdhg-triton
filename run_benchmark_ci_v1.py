"""Strict CI wrapper for benchmark_promoted_routing_v1.

Unlike the notebook-friendly entry point, this process exits non-zero when CUDA
is unavailable or the benchmark does not finish with a PASS report.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmark_promoted_routing_v1 import RAW_REPORT, main


def strict_main() -> int:
    code = main()
    if not Path(RAW_REPORT).exists():
        print("BENCHMARK_CI: FAIL (report missing)")
        return 2
    data = json.loads(Path(RAW_REPORT).read_text(encoding="utf-8"))
    if code != 0 or data.get("status") != "PASS":
        print(f"BENCHMARK_CI: FAIL ({data.get('status')})")
        return 3
    print("BENCHMARK_CI: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(strict_main())
