"""No-argument CPU validation runner for the autonomous CLOSE1 pack."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from pdhg_close1 import norm_k_periodic_2d, pdhg, primal_objective


HERE = Path(__file__).resolve().parent
OUT = HERE / "close1_cpu_validation_metrics.json"


def case_data(kind: str):
    g = torch.Generator().manual_seed(20260731)
    obs = torch.rand((1, 1, 31, 47), generator=g, dtype=torch.float64) + 0.4
    if kind == "gaussian":
        return obs + 0.2 * torch.randn(obs.shape, generator=g, dtype=obs.dtype), obs
    if kind == "poisson_log":
        return torch.log(obs), obs
    if kind in ("poisson_intensity", "kl"):
        return obs.clone(), obs
    if kind == "xlogx":
        return torch.ones_like(obs), None
    raise ValueError(kind)


def main() -> int:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=HERE,
        text=True, capture_output=True,
    )
    rows = []
    for kind in ("gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx"):
        x0, obs = case_data(kind)
        initial = float(primal_objective(kind, x0, obs, 0.06).item())
        x, dual, info = pdhg(
            kind, x0, obs, 0.06,
            max_iter=1200, min_iter=40, tol=1e-8, monitor_every=20,
            kkt_tol=2e-5, gap_tol=2e-5, backend="torch",
        )
        rows.append({
            "kind": kind,
            "initial_objective": initial,
            "final_objective": info.diagnostics.primal_objective,
            "objective_drop": initial - info.diagnostics.primal_objective,
            "iterations": info.iterations,
            "stabilized": info.stabilized,
            "certified": info.certified,
            "relative_gap": info.diagnostics.relative_gap,
            "kkt_max_residual": info.diagnostics.kkt_max_residual,
            "dual_feasibility_violation": info.diagnostics.dual_feasibility_violation,
            "finite_primal": bool(torch.isfinite(x).all().item()),
            "finite_dual": bool(torch.isfinite(dual).all().item()),
        })
    report = {
        "schema": "lambert-prox-close1-cpu-validation-v1",
        "status": "PASS" if proc.returncode == 0 and all(
            r["finite_primal"] and r["finite_dual"] and r["objective_drop"] >= -1e-10
            and r["stabilized"] and r["certified"]
            for r in rows
        ) else "FAIL",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "finite_grid_norm_31x47": norm_k_periodic_2d(31, 47),
        "pytest": {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        "models": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "stabilized is an iterate-change stop flag; certified is a numerical "
            "gap/KKT threshold flag, not a floating-point convergence proof."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
