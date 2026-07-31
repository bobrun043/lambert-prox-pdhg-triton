from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from routing_policy_v1 import select_backend


def main() -> int:
    key = "Test GPU|7.5|12.8|2.11.0+cu128"
    table = {
        "schema": "lambert-prox-pdhg-routing-table-v1",
        "status": "PASS",
        "policy": {"fallback": "torch"},
        "devices": {
            key: {
                "environment": {},
                "routes": {
                    "kl": [
                        {"height": 128, "width": 128, "pixels": 16384, "backend": "triton_elementwise"},
                        {"height": 512, "width": 512, "pixels": 262144, "backend": "triton_stencil"},
                    ]
                },
            }
        },
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "table.json"
        p.write_text(json.dumps(table), encoding="utf-8")
        with patch("routing_policy_v1.current_device_key", return_value=key):
            d = select_backend("kl", 140, 140, p)
            assert d.backend == "triton_elementwise"
            d = select_backend("kl", 500, 500, p)
            assert d.backend == "triton_stencil"
            d = select_backend("kl", 4096, 4096, p)
            assert d.backend == "torch"
            d = select_backend("gaussian", 128, 128, p)
            assert d.backend == "torch"
        bad = Path(td) / "bad.json"
        bad.write_text(json.dumps({"schema": "x", "status": "PASS"}), encoding="utf-8")
        d = select_backend("kl", 128, 128, bad)
        assert d.backend == "torch"
    print("ROUTING_POLICY_TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
