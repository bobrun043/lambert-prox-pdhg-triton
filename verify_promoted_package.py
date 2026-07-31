"""Verify the promoted Lambert-Prox PDHG package without requiring CUDA."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUIRED = [
    "lambert_prox_torch_v1.py",
    "pdhg_promoted_v2.py",
    "triton_prox_canonical_v1.py",
    "triton_stencil_canonical_v1.py",
    "triton_validation_metrics_v2.json",
    "promotion_manifest_v1.json",
]


def main() -> int:
    for name in REQUIRED:
        path = HERE / name
        if not path.exists():
            raise RuntimeError(f"missing promoted artifact: {name}")
    for name in REQUIRED[:4]:
        py_compile.compile(str(HERE / name), doraise=True)

    import sys
    sys.path.insert(0, str(HERE))
    from pdhg_promoted_v2 import verify_promotion_manifest

    manifest = verify_promotion_manifest(HERE / "promotion_manifest_v1.json")
    report = json.loads((HERE / "triton_validation_metrics_v2.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["summary"]["count"] == 47
    assert report["summary"]["passed"] == 47
    assert report["summary"]["worst_scaled"] <= 8.0e-5
    assert manifest["status"] == "PROMOTED_CORRECTNESS_ONLY"
    print("PROMOTED_PACKAGE_VERIFY: PASS")
    print(json.dumps({
        "validation_device": report["environment"]["device"],
        "tests": report["summary"]["count"],
        "worst_scaled": report["summary"]["worst_scaled"],
        "provenance_level": manifest["provenance"]["level"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
