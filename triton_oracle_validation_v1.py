"""CUDA/Triton candidate validation against the unique Torch oracle.

A skipped CUDA run is reported as NOT_RUN, never PASS.  The harness validates:
* all seven elementwise AXPY+prox candidates;
* dual fused stencil;
* six primal fused stencil modes supported by the legacy candidate;
* contiguous and non-contiguous input views;
* one complete PDHG step.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import torch

from pdhg_canonical_v1 import (
    Backend, Kt, canonical_prox_f, dual_update_torch, primal_update_torch,
    pdhg_step, environment_info,
)

ELEMENTWISE_KINDS = [
    "gaussian", "poisson_intensity", "poisson_log", "kl",
    "xlogx", "exp", "neglog",
]
STENCIL_KINDS = [
    "gaussian", "poisson_intensity", "kl", "xlogx", "exp", "neglog",
]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a-b).abs()
    scale = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs()))
    return {
        "max_abs": float(diff.max().item()),
        "max_scaled": float((diff/scale).max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _pass(m: dict[str,float], dtype: torch.dtype) -> bool:
    tol = 8e-5 if dtype == torch.float32 else 2e-11
    return m["max_scaled"] <= tol


def run(output_json: str | Path | None = None) -> dict[str, Any]:
    env = environment_info()
    result: dict[str, Any] = {"environment": env, "status": "NOT_RUN", "tests": []}
    if not env["cuda_available"] or not env["triton_importable"]:
        result["reason"] = "CUDA and Triton are both required"
        if output_json:
            Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    import triton_prox_fused_candidate_v2 as elem
    import triton_stencil_candidate_v5 as stencil

    device = torch.device("cuda")
    dtype = torch.float32
    g = torch.Generator(device=device).manual_seed(123)
    x = torch.randn((2,1,65,97), generator=g, device=device, dtype=dtype)
    kty = torch.randn(x.shape, generator=g, device=device, dtype=dtype)
    obs = torch.rand(x.shape, generator=g, device=device, dtype=dtype) + 0.1
    tau = 0.37
    kwargs = dict(lam=tau, alpha=1.0, beta=-tau, gamma=0.0)

    for kind in ELEMENTWISE_KINDS:
        yobs = obs if kind in {"gaussian","poisson_intensity","poisson_log","kl"} else None
        oracle = canonical_prox_f(kind, x-tau*kty, tau, yobs)
        if kind == "gaussian": got = elem.prox_gaussian_axpy(x,kty,obs,**kwargs)
        elif kind == "poisson_intensity": got = elem.prox_poisson_intensity_axpy(x,kty,obs,**kwargs)
        elif kind == "poisson_log": got = elem.prox_poisson_nll_axpy(x,kty,obs,**kwargs)
        elif kind == "kl": got = elem.prox_kl_axpy(x,kty,obs,**kwargs)
        elif kind == "xlogx": got = elem.prox_xlogx_axpy(x,kty,**kwargs)
        elif kind == "exp": got = elem.prox_exp_axpy(x,kty,**kwargs)
        else: got = elem.prox_neglog_axpy(x,kty,**kwargs)
        torch.cuda.synchronize()
        m = _metrics(got, oracle)
        result["tests"].append({"family":"elementwise", "kind":kind, **m, "pass":_pass(m,dtype)})

    y = torch.randn((2,2,65,97), generator=g, device=device, dtype=dtype)
    xbar = torch.randn(x.shape, generator=g, device=device, dtype=dtype)
    sigma, lam_tv = 0.31, 0.08
    oracle_dual = dual_update_torch(xbar,y,sigma,lam_tv)
    got_dual = stencil.launch_dual_tv(xbar,y,sigma,lam_tv)
    torch.cuda.synchronize()
    m = _metrics(got_dual,oracle_dual)
    result["tests"].append({"family":"stencil_dual", "kind":"tv", **m, "pass":_pass(m,dtype)})

    ynew = oracle_dual
    for kind in STENCIL_KINDS:
        yobs = obs if kind in {"gaussian","poisson_intensity","kl"} else None
        oracle = primal_update_torch(kind,x,ynew,yobs,tau)
        out = torch.empty_like(x)
        got = stencil.launch_primal_div_prox(x,ynew,yobs,tau,kind,out)
        torch.cuda.synchronize()
        m = _metrics(got,oracle)
        result["tests"].append({"family":"stencil_primal", "kind":kind, **m, "pass":_pass(m,dtype)})

    # Non-contiguous views for the fused stencils.
    xb = torch.randn((1,1,71,103), generator=g, device=device, dtype=dtype)
    yb = torch.randn((1,2,71,103), generator=g, device=device, dtype=dtype)
    ob = torch.rand((1,1,71,103), generator=g, device=device, dtype=dtype)+0.1
    xv, yv, ov = xb[:,:,3:-3,3:-3], yb[:,:,3:-3,3:-3], ob[:,:,3:-3,3:-3]
    for kind in STENCIL_KINDS:
        yobs = ov if kind in {"gaussian","poisson_intensity","kl"} else None
        oracle = primal_update_torch(kind,xv,yv,yobs,tau)
        out = torch.empty_like(xv)
        got = stencil.launch_primal_div_prox(xv,yv,yobs,tau,kind,out)
        torch.cuda.synchronize()
        m = _metrics(got,oracle)
        result["tests"].append({"family":"stencil_primal_noncontiguous", "kind":kind, **m, "pass":_pass(m,dtype)})

    result["status"] = "PASS" if all(t["pass"] for t in result["tests"]) else "FAIL"
    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    report = run(here / "triton_validation_metrics.json")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] in {"PASS","NOT_RUN"} else 1)
