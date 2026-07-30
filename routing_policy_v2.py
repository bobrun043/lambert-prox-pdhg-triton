"""Conservative runtime routing for canonical Lambert-Prox PDHG.

The table is device-local evidence, not a general performance model.  Automatic
routing requires an exact software/GPU environment key, CUDA FP32 tensors, a
validated PASS row, a robust bootstrap lower bound above the table threshold,
and a nearby tested image geometry.  Otherwise the policy explicitly returns
Torch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


DEFAULT_TABLE = Path(__file__).with_name("pdhg_routing_table_v1.json")
DEFAULT_EVIDENCE_MANIFEST = Path(__file__).with_name("routing_evidence_manifest_v1.json")
_ALLOWED_BACKENDS = {"torch", "triton_elementwise", "triton_stencil"}


class RoutingEvidenceError(RuntimeError):
    """Raised when the packaged routing evidence has been modified or is invalid."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RouteDecision:
    requested: str
    selected_backend: str
    reason: str
    environment_key: str
    kind: str
    height: int
    width: int
    matched_height: int | None = None
    matched_width: int | None = None
    area_ratio: float | None = None
    aspect_ratio_factor: float | None = None
    paired_median_speedup: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    exact_environment: bool = False
    evidence_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_device_key(device: int = 0) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    return "|".join([
        torch.cuda.get_device_name(device),
        f"{props.major}.{props.minor}",
        str(torch.version.cuda),
        str(torch.__version__),
    ])


def verify_routing_evidence(
    manifest_path: str | Path = DEFAULT_EVIDENCE_MANIFEST,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise RoutingEvidenceError(f"routing evidence manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "lambert-prox-routing-evidence-v1":
        raise RoutingEvidenceError(f"unexpected routing evidence schema: {data.get('schema')!r}")
    base = path.parent
    for filename, expected in data.get("files", {}).items():
        p = base / filename
        if not p.exists():
            raise RoutingEvidenceError(f"evidence file missing: {filename}")
        actual = sha256_file(p)
        if actual != expected:
            raise RoutingEvidenceError(f"evidence hash mismatch: {filename}")
    table_path = base / data["routing_table"]
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if table.get("schema") != "lambert-prox-pdhg-routing-table-v1" or table.get("status") != "PASS":
        raise RoutingEvidenceError("routing table is not an accepted PASS table")
    raw_path = base / data["raw_benchmark"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "lambert-prox-pdhg-routing-benchmark-v1" or raw.get("status") != "PASS":
        raise RoutingEvidenceError("raw benchmark is not an accepted PASS report")
    cases = raw.get("cases", [])
    if len(cases) != 10 or any(not c.get("routing_decision") for c in cases):
        raise RoutingEvidenceError("raw benchmark does not contain the expected 10 routed cases")
    threshold = float(table.get("policy", {}).get("minimum_robust_speedup", 1.1))
    if threshold < 1.0:
        raise RoutingEvidenceError("invalid minimum robust speedup")
    for case in cases:
        ci_low = float(case["routing_decision"]["speedup_vs_torch"]["ci95_low"])
        if ci_low < threshold:
            raise RoutingEvidenceError("a promoted benchmark row violates the robust speedup threshold")
    return data


def _load_table(table_path: str | Path) -> dict[str, Any]:
    path = Path(table_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "lambert-prox-pdhg-routing-table-v1" or data.get("status") != "PASS":
        raise RoutingEvidenceError("routing table is not a validated PASS table")
    return data


def _aspect_ratio(height: int, width: int) -> float:
    return max(height / width, width / height)


def select_backend_for_environment(
    kind: str,
    height: int,
    width: int,
    *,
    environment_key: str,
    device_type: str,
    dtype: torch.dtype,
    table_path: str | Path = DEFAULT_TABLE,
    evidence_manifest: str | Path = DEFAULT_EVIDENCE_MANIFEST,
    max_area_ratio: float | None = None,
    max_aspect_ratio_factor: float = 2.0,
    verify_evidence: bool = True,
) -> RouteDecision:
    k = kind.lower().strip()
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_aspect_ratio_factor < 1.0:
        raise ValueError("max_aspect_ratio_factor must be >= 1")
    evidence_ok = False
    if verify_evidence:
        verify_routing_evidence(evidence_manifest)
        evidence_ok = True
    if device_type != "cuda":
        return RouteDecision("auto", "torch", "automatic Triton routing requires CUDA", environment_key, k, height, width, evidence_verified=evidence_ok)
    if dtype != torch.float32:
        return RouteDecision("auto", "torch", "promoted routing is validated for float32 only", environment_key, k, height, width, evidence_verified=evidence_ok)
    table = _load_table(table_path)
    device = table.get("devices", {}).get(environment_key)
    if device is None:
        return RouteDecision("auto", "torch", "no exact validated software/GPU environment", environment_key, k, height, width, evidence_verified=evidence_ok)
    entries = device.get("routes", {}).get(k, [])
    if not entries:
        return RouteDecision("auto", "torch", f"no validated route for model {k!r}", environment_key, k, height, width, exact_environment=True, evidence_verified=evidence_ok)
    pixels = height * width
    nearest = min(entries, key=lambda e: abs(math.log(max(int(e["pixels"]), 1) / pixels)))
    tested_pixels = int(nearest["pixels"])
    area_ratio = max(tested_pixels / pixels, pixels / tested_pixels)
    table_limit = float(table.get("policy", {}).get("interpolation_max_area_ratio", 4.0))
    if "interpolation_max_area_ratio" not in table.get("policy", {}):
        table_limit = 4.0
    area_limit = table_limit if max_area_ratio is None else min(float(max_area_ratio), table_limit)
    if area_ratio > area_limit:
        return RouteDecision(
            "auto", "torch", f"nearest tested area is too distant ({area_ratio:.2f}x > {area_limit:.2f}x)",
            environment_key, k, height, width,
            int(nearest["height"]), int(nearest["width"]), float(area_ratio),
            exact_environment=True, evidence_verified=evidence_ok,
        )
    current_aspect = _aspect_ratio(height, width)
    tested_aspect = _aspect_ratio(int(nearest["height"]), int(nearest["width"]))
    aspect_factor = max(current_aspect / tested_aspect, tested_aspect / current_aspect)
    if aspect_factor > max_aspect_ratio_factor:
        return RouteDecision(
            "auto", "torch", f"aspect ratio is too distant ({aspect_factor:.2f}x > {max_aspect_ratio_factor:.2f}x)",
            environment_key, k, height, width,
            int(nearest["height"]), int(nearest["width"]), float(area_ratio), float(aspect_factor),
            exact_environment=True, evidence_verified=evidence_ok,
        )
    backend = str(nearest.get("backend", "torch"))
    if backend not in _ALLOWED_BACKENDS:
        return RouteDecision("auto", "torch", f"invalid backend in routing table: {backend!r}", environment_key, k, height, width, exact_environment=True, evidence_verified=evidence_ok)
    decision = nearest.get("decision", {})
    speed = decision.get("speedup_vs_torch", {})
    threshold = float(table.get("policy", {}).get("minimum_robust_speedup", 1.1))
    ci_low = float(speed.get("ci95_low", 0.0))
    if backend != "torch" and ci_low < threshold:
        return RouteDecision("auto", "torch", f"robust speedup evidence below threshold ({ci_low:.3f}x < {threshold:.3f}x)", environment_key, k, height, width, exact_environment=True, evidence_verified=evidence_ok)
    return RouteDecision(
        "auto", backend, "nearest validated geometry on exact environment; robust bootstrap threshold passed",
        environment_key, k, height, width,
        int(nearest["height"]), int(nearest["width"]), float(area_ratio), float(aspect_factor),
        float(speed.get("paired_median_speedup")), ci_low, float(speed.get("ci95_high")),
        True, evidence_ok,
    )


def select_backend(
    kind: str,
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    table_path: str | Path = DEFAULT_TABLE,
    evidence_manifest: str | Path = DEFAULT_EVIDENCE_MANIFEST,
    max_area_ratio: float | None = None,
    max_aspect_ratio_factor: float = 2.0,
) -> RouteDecision:
    dev = torch.device(device)
    return select_backend_for_environment(
        kind, height, width,
        environment_key=current_device_key(dev.index or 0) if dev.type == "cuda" else "cpu",
        device_type=dev.type,
        dtype=dtype,
        table_path=table_path,
        evidence_manifest=evidence_manifest,
        max_area_ratio=max_area_ratio,
        max_aspect_ratio_factor=max_aspect_ratio_factor,
    )


__all__ = [
    "DEFAULT_TABLE", "DEFAULT_EVIDENCE_MANIFEST", "RoutingEvidenceError", "RouteDecision",
    "sha256_file", "current_device_key", "verify_routing_evidence",
    "select_backend_for_environment", "select_backend",
]
