from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import lambert_prox_reference_v1 as ref
import lambert_prox_torch_v1 as tt


HERE = Path(__file__).resolve().parent


def main() -> int:
    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "test_lambert_prox_torch_v1.py"]
    proc = subprocess.run(pytest_cmd, cwd=HERE, text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    metrics: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_executed": "cpu",
        "pytest_returncode": proc.returncode,
        "pytest_stdout": proc.stdout,
    }

    for dtype, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
        R = torch.cat(
            [
                torch.tensor([-1000.0, -800.0, -100.0, -20.0, -8.0001, -8.0, -7.9999], dtype=dtype),
                torch.linspace(-7.5, 40.0, 100001, dtype=dtype),
                torch.logspace(2.0, 6.0, 10000, dtype=dtype),
            ]
        )
        out = tt.solve_u_log_u_pair(R)
        oracle = ref.solve_u_log_u_reference(R.double().numpy())
        q_ref = torch.from_numpy(oracle.q).to(dtype)
        q_rel = torch.abs(out.q - q_ref) / torch.maximum(torch.ones_like(q_ref), torch.abs(q_ref))
        normalized_residual = out.residual / torch.maximum(torch.ones_like(R), torch.abs(R))
        metrics[name] = {
            "points": int(R.numel()),
            "all_converged": bool(torch.all(out.converged)),
            "log_only": int(out.log_only.sum()),
            "max_residual_absolute": float(out.residual.max()),
            "p99_residual_absolute": float(torch.quantile(out.residual, 0.99)),
            "max_residual_normalized": float(normalized_residual.max()),
            "max_q_relative_vs_oracle_rounded": float(q_rel.max()),
        }

    def norm_res(res: torch.Tensor, scale: torch.Tensor) -> float:
        return float((torch.abs(res) / torch.maximum(torch.ones_like(scale), scale)).max())

    # KKT summary on the same broad randomized domain for both dtypes.
    for dtype, key in [(torch.float64, "kkt_fp64_max_normalized"), (torch.float32, "kkt_fp32_max_normalized")]:
        gen = torch.Generator().manual_seed(20260730)
        v = torch.randn(50000, dtype=dtype, generator=gen) * 5.0
        lam = torch.exp(torch.empty(50000, dtype=dtype).uniform_(-4.0, 4.0, generator=gen))
        y = torch.exp(torch.empty(50000, dtype=dtype).uniform_(-5.0, 5.0, generator=gen))
        kkt: dict[str, float] = {}
        x = tt.prox_exp(v, lam)
        kkt["exp"] = norm_res(x-v+lam*torch.exp(x), torch.abs(v)+torch.abs(x)+lam*torch.exp(x))
        x = tt.prox_xlogx(v, lam)
        term = lam*(1+tt.prox_xlogx_log(v, lam)); kkt["xlogx"] = norm_res(x-v+term, torch.abs(v)+torch.abs(x)+torch.abs(term))
        x = tt.prox_kl(v, y, lam)
        term = lam*(tt.prox_kl_log(v, y, lam)-torch.log(y)); kkt["kl"] = norm_res(x-v+term, torch.abs(v)+torch.abs(x)+torch.abs(term))
        x = tt.prox_poisson_log(v, y, lam)
        term = lam*(torch.exp(x)-y); kkt["poisson_log"] = norm_res(x-v+term, torch.abs(v)+torch.abs(x)+lam*(torch.exp(x)+y))
        x = tt.prox_poisson_intensity(v, y, lam)
        term = lam*(1-y/x); kkt["poisson_intensity"] = norm_res(x-v+term, torch.abs(v)+torch.abs(x)+lam*(1+y/x))
        x = tt.prox_neglog(v, lam)
        kkt["neglog"] = norm_res(x-v-lam/x, torch.abs(v)+torch.abs(x)+lam/x)
        yg = torch.randn(50000, dtype=dtype, generator=gen)
        x = tt.prox_gaussian(v, yg, lam)
        kkt["gaussian"] = norm_res(x-v+lam*(x-yg), torch.abs(v)+torch.abs(x)+lam*torch.abs(x-yg))
        metrics[key] = kkt

    metrics_path = HERE / "torch_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
