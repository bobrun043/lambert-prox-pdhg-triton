"""No-arg CUDA regression for the promoted PDHG backends.

This script is a correctness comparison, not a performance claim. It executes
five well-posed PDHG models through Torch, Triton elementwise and Triton fused
stencil backends, then writes a JSON report.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPORT = HERE / "pdhg_promoted_regression_metrics.json"

from pdhg_promoted_v2 import Backend, objective_value, pdhg, verify_promotion_manifest


def scaled_max(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs()))
    return float(((a - b).abs() / den).max().item())


def timed_run(*args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pdhg(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> int:
    manifest_path = HERE / "promotion_manifest_v1.json"
    manifest = verify_promotion_manifest(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Triton is required: {exc}") from exc

    torch.manual_seed(20260730)
    device = torch.device("cuda")
    dtype = torch.float32
    shape = (1, 1, 128, 192)

    cases = []
    y = torch.rand(shape, device=device, dtype=dtype) + 0.1
    cases.append(("gaussian", y.clone(), y, 0.08, 120))
    y = torch.rand(shape, device=device, dtype=dtype) * 4.0 + 0.05
    cases.append(("poisson_intensity", y.clone(), y, 0.04, 180))
    y = torch.rand(shape, device=device, dtype=dtype) * 3.0 + 0.1
    cases.append(("poisson_log", torch.log(y), y, 0.04, 160))
    y = torch.rand(shape, device=device, dtype=dtype) + 0.1
    cases.append(("kl", y.clone(), y, 0.03, 180))
    x0 = torch.rand(shape, device=device, dtype=dtype) + 0.3
    cases.append(("xlogx", x0, None, 0.01, 100))

    rows = []
    for kind, x0, y_obs, lam_tv, iters in cases:
        common = dict(
            kind=kind, x0=x0, y_obs=y_obs, lam_tv=lam_tv,
            max_iter=iters, min_iter=iters, tol=0.0,
            monitor_every=max(1, iters // 4),
            promotion_manifest=HERE / "promotion_manifest_v1.json",
        )
        (x_t, y_t, info_t), time_t = timed_run(**common, backend=Backend.TORCH)
        (x_e, y_e, info_e), time_e = timed_run(**common, backend=Backend.TRITON_ELEMENTWISE)
        (x_s, y_s, info_s), time_s = timed_run(**common, backend=Backend.TRITON_STENCIL)

        row = {
            "kind": kind,
            "iterations": iters,
            "torch_seconds": time_t,
            "triton_elementwise_seconds": time_e,
            "triton_stencil_seconds": time_s,
            "elementwise_primal_scaled": scaled_max(x_e, x_t),
            "elementwise_dual_scaled": scaled_max(y_e, y_t),
            "stencil_primal_scaled": scaled_max(x_s, x_t),
            "stencil_dual_scaled": scaled_max(y_s, y_t),
            "objective_torch": float(objective_value(kind, x_t, y_obs, lam_tv).item()),
            "objective_elementwise": float(objective_value(kind, x_e, y_obs, lam_tv).item()),
            "objective_stencil": float(objective_value(kind, x_s, y_obs, lam_tv).item()),
            "promotion_elementwise": info_e.backend.promotion_verified,
            "promotion_stencil": info_s.backend.promotion_verified,
        }
        row["pass"] = max(
            row["elementwise_primal_scaled"], row["elementwise_dual_scaled"],
            row["stencil_primal_scaled"], row["stencil_dual_scaled"],
        ) <= 8.0e-5
        rows.append(row)
        print(kind, "PASS" if row["pass"] else "FAIL", row)

    result = {
        "schema": "lambert-prox-pdhg-promoted-regression-v1",
        "status": "PASS" if all(r["pass"] for r in rows) else "FAIL",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "scope": "correctness regression; timings are descriptive and not a speedup claim",
        "promotion_evidence": {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "sources": manifest["sources"],
            "validation_report_sha256": manifest["validation_report"]["sha256"],
        },
        "rows": rows,
    }
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("promoted PDHG regression failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
