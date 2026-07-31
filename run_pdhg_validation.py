from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from pdhg_canonical_v1 import (
    check_adjoint, environment_info, objective_value, pdhg,
)
from triton_oracle_validation_v1 import run as run_triton


def _case_data(kind: str, shape=(1,1,40,53), dtype=torch.float64):
    g = torch.Generator().manual_seed(20260730)
    obs = torch.rand(shape, generator=g, dtype=dtype) + 0.4
    if kind == "gaussian":
        x0 = obs + 0.35 * torch.randn(shape, generator=g, dtype=dtype)
        return x0, obs
    if kind == "poisson_intensity":
        return obs.clone(), obs
    if kind == "poisson_log":
        return torch.log(obs), obs
    if kind == "kl":
        return torch.ones_like(obs), obs
    if kind == "xlogx":
        return torch.ones_like(obs), None
    raise ValueError(kind)


def main() -> int:
    here = Path(__file__).resolve().parent
    env = environment_info()
    metrics = {
        "schema": "lambert-prox-pdhg-validation-v1",
        "environment": env,
        "adjoint_relative_gap_fp64": check_adjoint(),
        "pytest": {},
        "torch_pdhg": [],
        "triton": {},
    }

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=here,
        text=True,
        capture_output=True,
    )
    metrics["pytest"] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": time.perf_counter() - t0,
    }

    for kind in ["gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx"]:
        x0, obs = _case_data(kind)
        lam_tv = 0.06
        e0 = float(objective_value(kind, x0, obs, lam_tv).item())
        x, y, info = pdhg(
            kind, x0, obs, lam_tv,
            max_iter=600,
            min_iter=40,
            tol=2e-6,
            backend="torch",
            monitor_every=20,
        )
        ef = float(objective_value(kind, x, obs, lam_tv).item())
        metrics["torch_pdhg"].append({
            "kind": kind,
            "objective_initial": e0,
            "objective_final": ef,
            "objective_drop": e0 - ef,
            "iterations": info.iterations,
            "converged": info.converged,
            "primal_rel_change": info.primal_rel_change,
            "dual_rel_change": info.dual_rel_change,
            "elapsed_seconds": info.elapsed_seconds,
            "backend_requested": info.backend.requested,
            "backend_executed": info.backend.executed,
            "finite_primal": bool(torch.isfinite(x).all().item()),
            "finite_dual": bool(torch.isfinite(y).all().item()),
        })

    metrics["triton"] = run_triton(here / "triton_validation_metrics.json")
    out = here / "pdhg_validation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    torch_ok = (
        proc.returncode == 0
        and metrics["adjoint_relative_gap_fp64"] < 5e-14
        and all(r["finite_primal"] and r["finite_dual"] and r["objective_drop"] >= -1e-10
                and r["backend_executed"] == "torch" for r in metrics["torch_pdhg"])
    )
    triton_ok = metrics["triton"]["status"] in {"PASS", "NOT_RUN"}
    return 0 if torch_ok and triton_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
