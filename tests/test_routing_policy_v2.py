from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

from routing_policy_v2 import (
    DEFAULT_EVIDENCE_MANIFEST,
    DEFAULT_TABLE,
    RoutingEvidenceError,
    select_backend_for_environment,
    verify_routing_evidence,
)

T4_KEY = "Tesla T4|7.5|12.8|2.11.0+cu128"
KINDS = ["gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx"]


def decide(kind, h, w, **kw):
    return select_backend_for_environment(
        kind, h, w, environment_key=T4_KEY, device_type="cuda", dtype=torch.float32,
        table_path=DEFAULT_TABLE, evidence_manifest=DEFAULT_EVIDENCE_MANIFEST, **kw,
    )


def run() -> int:
    assertions = 0
    verify_routing_evidence(); assertions += 1
    for kind in KINDS:
        for h, w in [(128, 192), (512, 512)]:
            d = decide(kind, h, w)
            assert d.selected_backend == "triton_stencil" and d.ci95_low >= 1.10
            assertions += 1
    d = decide("kl", 256, 256)
    assert d.selected_backend == "triton_stencil" and (d.matched_height, d.matched_width) == (128, 192); assertions += 1
    d = decide("poisson_log", 1024, 1024)
    assert d.selected_backend == "triton_stencil" and abs(d.area_ratio - 4.0) < 1e-12; assertions += 1
    assert decide("kl", 64, 64).selected_backend == "torch"; assertions += 1
    assert decide("kl", 16, 1536).selected_backend == "torch"; assertions += 1
    d = select_backend_for_environment(
        "kl", 128, 192, environment_key="Unknown GPU|0.0|0|0", device_type="cuda", dtype=torch.float32,
        table_path=DEFAULT_TABLE, evidence_manifest=DEFAULT_EVIDENCE_MANIFEST,
    )
    assert d.selected_backend == "torch"; assertions += 1
    d = select_backend_for_environment(
        "kl", 128, 192, environment_key=T4_KEY, device_type="cpu", dtype=torch.float32,
        table_path=DEFAULT_TABLE, evidence_manifest=DEFAULT_EVIDENCE_MANIFEST,
    )
    assert d.selected_backend == "torch"; assertions += 1
    d = select_backend_for_environment(
        "kl", 128, 192, environment_key=T4_KEY, device_type="cuda", dtype=torch.float64,
        table_path=DEFAULT_TABLE, evidence_manifest=DEFAULT_EVIDENCE_MANIFEST,
    )
    assert d.selected_backend == "torch"; assertions += 1
    assert decide("unsupported", 128, 192).selected_backend == "torch"; assertions += 1
    manifest = json.loads(Path(DEFAULT_EVIDENCE_MANIFEST).read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for filename in manifest["files"]:
            (td / filename).write_bytes((Path(DEFAULT_EVIDENCE_MANIFEST).parent / filename).read_bytes())
        bad = copy.deepcopy(manifest)
        bad["files"]["pdhg_routing_table_v1.json"] = "0" * 64
        (td / "routing_evidence_manifest_v1.json").write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_routing_evidence(td / "routing_evidence_manifest_v1.json")
        except RoutingEvidenceError:
            assertions += 1
        else:
            raise AssertionError("tampered evidence was accepted")
    print(f"ROUTING_POLICY_V2: PASS ({assertions} assertion groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
