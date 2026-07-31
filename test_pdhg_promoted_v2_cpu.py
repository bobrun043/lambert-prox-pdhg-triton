"""CPU regression tests for the promoted PDHG integration.

These tests validate the canonical Torch path and the promotion evidence. They
cannot execute the promoted CUDA kernels in a CPU-only environment.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pdhg_promoted_v2 import (
    Backend,
    BackendUnavailableError,
    check_adjoint,
    objective_value,
    pdhg,
    resolve_backend,
    verify_promotion_manifest,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    passed = 0
    evidence = verify_promotion_manifest(HERE / "promotion_manifest_v1.json")
    _assert(evidence["status"] == "PROMOTED_CORRECTNESS_ONLY", "promotion status")
    passed += 1

    gap = check_adjoint((2, 1, 31, 47), dtype=torch.float64, trials=8)
    _assert(gap < 1e-12, f"adjoint gap {gap}")
    passed += 1

    g = torch.Generator().manual_seed(42)
    shape = (1, 1, 31, 47)
    cases = []

    y = torch.rand(shape, generator=g, dtype=torch.float64) + 0.1
    cases.append(("gaussian", y.clone(), y, 0.08, 120))

    y = torch.rand(shape, generator=g, dtype=torch.float64) * 4.0 + 0.05
    cases.append(("poisson_intensity", y.clone(), y, 0.04, 180))

    y = torch.rand(shape, generator=g, dtype=torch.float64) * 3.0 + 0.1
    cases.append(("poisson_log", torch.log(y), y, 0.04, 160))

    y = torch.rand(shape, generator=g, dtype=torch.float64) + 0.1
    cases.append(("kl", y.clone(), y, 0.03, 180))

    x0 = torch.rand(shape, generator=g, dtype=torch.float64) + 0.3
    cases.append(("xlogx", x0, None, 0.01, 100))

    for kind, x0, y_obs, lam_tv, iterations in cases:
        initial = float(objective_value(kind, x0, y_obs, lam_tv).item())
        x, dual, info = pdhg(
            kind, x0, y_obs, lam_tv,
            max_iter=iterations, min_iter=iterations, tol=0.0,
            backend=Backend.TORCH, monitor_every=max(1, iterations // 4),
        )
        final = float(objective_value(kind, x, y_obs, lam_tv).item())
        _assert(torch.isfinite(x).all().item(), f"{kind}: nonfinite primal")
        _assert(torch.isfinite(dual).all().item(), f"{kind}: nonfinite dual")
        _assert(math.isfinite(final), f"{kind}: nonfinite objective")
        _assert(final <= initial + 1e-9 * max(1.0, abs(initial)),
                f"{kind}: objective increased {initial} -> {final}")
        _assert(info.backend.executed == "torch", f"{kind}: wrong backend")
        passed += 3

    for invalid_kind in ("exp", "neglog"):
        try:
            pdhg(invalid_kind, torch.ones(shape, dtype=torch.float64), None, 0.1,
                 max_iter=2, min_iter=2, backend=Backend.TORCH)
        except ValueError:
            passed += 1
        else:
            raise AssertionError(f"{invalid_kind}+TV should be rejected")

    if not torch.cuda.is_available():
        try:
            resolve_backend(Backend.TRITON_STENCIL,
                            promotion_manifest=HERE / "promotion_manifest_v1.json")
        except BackendUnavailableError:
            passed += 1
        else:
            raise AssertionError("Triton backend silently accepted without CUDA")

    print(f"PDHG_PROMOTED_V2_CPU: PASS ({passed} assertions groups)")
    print(f"adjoint_gap={gap:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
