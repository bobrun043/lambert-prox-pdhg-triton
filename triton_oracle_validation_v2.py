"""Strict CUDA validation of canonical Triton candidates against Torch oracle.

NOT_RUN is distinct from PASS.  Use ``--require-cuda`` in CI to make missing
CUDA/Triton a non-zero result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import torch

from pdhg_canonical_v1 import (
    Kt,
    canonical_prox_f,
    dual_update_torch,
    primal_update_torch,
)

KINDS = [
    "gaussian", "poisson_intensity", "poisson_log", "kl",
    "xlogx", "exp", "neglog",
]
OBS_KINDS = {"gaussian", "poisson_intensity", "poisson_log", "kl"}


def environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "triton_importable": importlib.util.find_spec("triton") is not None,
    }


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (got - ref).abs()
    scale = torch.maximum(torch.ones_like(ref), torch.maximum(got.abs(), ref.abs()))
    return {
        "max_abs": float(diff.max().item()),
        "max_scaled": float((diff / scale).max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def passed(m: dict[str, float], tol: float = 8.0e-5) -> bool:
    return m["max_scaled"] <= tol


def append_test(rows: list[dict[str, Any]], family: str, kind: str,
                got: torch.Tensor, ref: torch.Tensor, extra: dict[str, Any] | None = None):
    torch.cuda.synchronize()
    m = metrics(got, ref)
    row = {"family": family, "kind": kind, **m, "pass": passed(m)}
    if extra:
        row.update(extra)
    rows.append(row)


def elementwise_call(mod, kind, x, kty, obs, tau):
    kw = dict(lam=tau, alpha=1.0, beta=-tau, gamma=0.0)
    if kind == "gaussian":
        return mod.prox_gaussian_axpy(x, kty, obs, **kw)
    if kind == "poisson_intensity":
        return mod.prox_poisson_intensity_axpy(x, kty, obs, **kw)
    if kind == "poisson_log":
        return mod.prox_poisson_log_axpy(x, kty, obs, **kw)
    if kind == "kl":
        return mod.prox_kl_axpy(x, kty, obs, **kw)
    if kind == "xlogx":
        return mod.prox_xlogx_axpy(x, kty, **kw)
    if kind == "exp":
        return mod.prox_exp_axpy(x, kty, **kw)
    return mod.prox_neglog_axpy(x, kty, **kw)


def run(output_json: str | Path | None = None) -> dict[str, Any]:
    env = environment()
    result: dict[str, Any] = {"schema": "triton-oracle-validation-v2", "environment": env,
                              "status": "NOT_RUN", "tests": []}
    if not (env["cuda_available"] and env["triton_importable"]):
        result["reason"] = "CUDA and Triton are both required"
        if output_json:
            Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    import triton_prox_canonical_v1 as elem
    import triton_stencil_canonical_v1 as stencil

    device = torch.device("cuda")
    dtype = torch.float32
    g = torch.Generator(device=device).manual_seed(20260730)
    tau, sigma, lam_tv = 0.37, 0.31, 0.08
    rows: list[dict[str, Any]] = result["tests"]

    # Standard odd, non-power-of-two geometry.
    x = torch.randn((2, 1, 65, 97), generator=g, device=device, dtype=dtype)
    kty = torch.randn_like(x, generator=g)
    obs = torch.rand_like(x, generator=g) + 0.1
    for kind in KINDS:
        yobs = obs if kind in OBS_KINDS else None
        ref = canonical_prox_f(kind, x - tau * kty, tau, yobs)
        got = elementwise_call(elem, kind, x, kty, yobs, tau)
        append_test(rows, "elementwise_standard", kind, got, ref)

    # Exact KL zero boundary: no epsilon regularization is accepted.
    obs_zero = obs.clone()
    obs_zero[..., ::3, ::5] = 0.0
    ref = canonical_prox_f("kl", x - tau * kty, tau, obs_zero)
    got = elementwise_call(elem, "kl", x, kty, obs_zero, tau)
    append_test(rows, "elementwise_boundary", "kl_y_zero", got, ref,
                {"exact_zero_count": int((got == 0).sum().item())})

    # Wide cancellation-sensitive regimes for exp and Poisson log.
    wide_v = torch.linspace(-100.0, 100.0, 200_003, device=device, dtype=dtype)
    wide_x = wide_v.reshape(1, 1, 1, -1)
    wide_k = torch.zeros_like(wide_x)
    wide_obs = torch.linspace(0.0, 50.0, wide_x.numel(), device=device, dtype=dtype).reshape_as(wide_x)
    for kind in ("exp", "poisson_log"):
        yobs = wide_obs if kind == "poisson_log" else None
        ref = canonical_prox_f(kind, wide_x, tau, yobs)
        got = elementwise_call(elem, kind, wide_x, wide_k, yobs, tau)
        append_test(rows, "elementwise_wide", kind, got, ref)

    # Non-contiguous elementwise views.
    bx = torch.randn((1, 1, 71, 103), generator=g, device=device, dtype=dtype)
    bk = torch.randn_like(bx, generator=g)
    bo = torch.rand_like(bx, generator=g) + 0.1
    xv, kv, ov = bx[:, :, 3:-3, 3:-3], bk[:, :, 3:-3, 3:-3], bo[:, :, 3:-3, 3:-3]
    assert not xv.is_contiguous()
    for kind in KINDS:
        yobs = ov if kind in OBS_KINDS else None
        ref = canonical_prox_f(kind, xv - tau * kv, tau, yobs)
        got = elementwise_call(elem, kind, xv, kv, yobs, tau)
        append_test(rows, "elementwise_noncontiguous", kind, got, ref)

    # Fused dual and all seven fused primal modes.
    y = torch.randn((2, 2, 65, 97), generator=g, device=device, dtype=dtype)
    xbar = torch.randn_like(x, generator=g)
    ref_dual = dual_update_torch(xbar, y, sigma, lam_tv)
    got_dual = stencil.launch_dual_tv(xbar, y, sigma, lam_tv)
    append_test(rows, "stencil_dual", "tv", got_dual, ref_dual)
    for kind in KINDS:
        yobs = obs if kind in OBS_KINDS else None
        ref = primal_update_torch(kind, x, ref_dual, yobs, tau)
        got = stencil.launch_primal_div_prox(x, ref_dual, yobs, tau, kind)
        append_test(rows, "stencil_primal", kind, got, ref)

    # Non-contiguous fused stencils.
    by = torch.randn((1, 2, 71, 103), generator=g, device=device, dtype=dtype)
    yv = by[:, :, 3:-3, 3:-3]
    ref_dual_v = dual_update_torch(xv, yv, sigma, lam_tv)
    got_dual_v = stencil.launch_dual_tv(xv, yv, sigma, lam_tv)
    append_test(rows, "stencil_dual_noncontiguous", "tv", got_dual_v, ref_dual_v)
    for kind in KINDS:
        yobs = ov if kind in OBS_KINDS else None
        ref = primal_update_torch(kind, xv, ref_dual_v, yobs, tau)
        got = stencil.launch_primal_div_prox(xv, ref_dual_v, yobs, tau, kind)
        append_test(rows, "stencil_primal_noncontiguous", kind, got, ref)

    # Complete one-step comparison for all proxes.
    for kind in KINDS:
        yobs = obs if kind in OBS_KINDS else None
        ref_y = dual_update_torch(xbar, y, sigma, lam_tv)
        ref_x = primal_update_torch(kind, x, ref_y, yobs, tau)
        got_y = stencil.launch_dual_tv(xbar, y, sigma, lam_tv)
        got_x = stencil.launch_primal_div_prox(x, got_y, yobs, tau, kind)
        append_test(rows, "pdhg_complete_step_dual", kind, got_y, ref_y)
        append_test(rows, "pdhg_complete_step_primal", kind, got_x, ref_x)

    result["status"] = "PASS" if all(row["pass"] for row in rows) else "FAIL"
    result["summary"] = {
        "count": len(rows),
        "passed": sum(bool(r["pass"]) for r in rows),
        "worst_scaled": max(r["max_scaled"] for r in rows),
    }
    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="triton_validation_metrics_v2.json")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    report = run(args.output)
    print(json.dumps(report, indent=2))
    if report["status"] == "PASS":
        return 0
    if report["status"] == "NOT_RUN":
        return 3 if args.require_cuda else 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
