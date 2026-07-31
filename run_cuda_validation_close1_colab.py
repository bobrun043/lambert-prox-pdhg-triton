"""No-argument T4 validation with source hashes captured before CUDA work.

Run this file from the extracted CLOSE1 directory in a Tesla T4 Colab.  The
script hashes every source involved in the oracle comparison before importing
the validation module, executes the 47 comparisons, hashes the sources again,
and accepts PASS only when the two maps are identical.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRE_RUN = HERE / "pre_run_provenance_close1.json"
REPORT = HERE / "triton_validation_metrics_close1.json"
SOURCES = (
    "lambert_prox_torch_v1.py",
    "pdhg_canonical_v1.py",
    "pdhg_promoted_v2.py",
    "triton_prox_canonical_v1.py",
    "triton_stencil_canonical_v1.py",
    "triton_oracle_validation_v2.py",
    "triton_promotion_gate_v1.py",
    "run_cuda_validation_close1_colab.py",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_map() -> dict[str, str]:
    missing = [name for name in SOURCES if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing CLOSE1 validation sources: {missing}")
    return {name: sha256_file(HERE / name) for name in SOURCES}


def release_digest(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    before = source_map()
    begun = datetime.now(timezone.utc).isoformat()
    pre = {
        "schema": "lambert-prox-close1-pre-run-provenance-v1",
        "captured_before_validation_import": True,
        "started_at_utc": begun,
        "release_sha256": release_digest(before),
        "sources": before,
    }
    PRE_RUN.write_text(json.dumps(pre, indent=2), encoding="utf-8")

    validator = importlib.import_module("triton_oracle_validation_v2")
    inner_path = HERE / "triton_validation_metrics_v2_close1_inner.json"
    inner = validator.run(inner_path)
    after = source_map()
    unchanged = before == after
    status = "PASS" if inner.get("status") == "PASS" and unchanged else inner.get("status", "FAIL")
    if inner.get("status") == "PASS" and not unchanged:
        status = "FAIL"
    report = {
        "schema": "lambert-prox-triton-validation-close1-v1",
        "status": status,
        "started_at_utc": begun,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "level": "pre_run_and_post_run_source_binding",
            "release_sha256": release_digest(before),
            "source_hashes_before": before,
            "source_hashes_after": after,
            "sources_unchanged_during_run": unchanged,
        },
        "validation": inner,
        "interpretation": (
            "PASS binds this exact source map to this execution. It does not extend "
            "the device claim beyond the environment recorded by the inner report."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else (3 if status == "NOT_RUN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
