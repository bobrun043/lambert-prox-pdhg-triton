"""CLOSE1 certification layer for the promoted Lambert-Prox PDHG solver.

The promoted V2 solver and the T4 evidence bound to its byte hash are kept
immutable.  This module calls that solver, supplies the exact finite-grid norm
when possible, and computes independent end-of-run diagnostics:

* relative primal and dual iterate changes (stabilization only),
* primal and dual objectives and their Fenchel gap,
* normalized primal and dual resolvent/KKT residuals,
* pointwise dual-feasibility violation.

``certified`` is a numerical acceptance flag for user-selected tolerances.  It
is not a proof of convergence of finite precision or inexact-prox iterations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch

import pdhg_promoted_v2 as legacy


@dataclass(frozen=True)
class Close1Diagnostics:
    primal_objective: float
    dual_objective: float
    primal_dual_gap: float
    relative_gap: float
    kkt_primal_residual: float
    kkt_dual_residual: float
    kkt_max_residual: float
    dual_feasibility_violation: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDHGInfoClose1:
    iterations: int
    stabilized: bool
    certified: bool
    stop_reason: str
    tau: float
    sigma: float
    norm_k_exact: float
    norm_k_step_bound: float
    backend: legacy.BackendReport
    primal_rel_change: float
    dual_rel_change: float
    diagnostics: Close1Diagnostics
    solver_elapsed_seconds: float
    diagnostics_elapsed_seconds: float
    history: dict[str, list[float]]
    status_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def norm_k_upper_bound_periodic_2d() -> float:
    """Dimension-independent safe bound for the periodic forward gradient."""
    return math.sqrt(8.0)


def norm_k_periodic_2d(height: int, width: int) -> float:
    """Exact l2 operator norm of the periodic 2-D forward-difference gradient.

    For a H x W grid, Fourier diagonalization gives

        ||K||^2 = 4 sin^2(pi floor(W/2)/W)
                + 4 sin^2(pi floor(H/2)/H).

    The value is sqrt(8) when both dimensions are even and is strictly smaller
    when at least one nontrivial dimension is odd.
    """
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("height and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    wx = 4.0 * math.sin(math.pi * (width // 2) / width) ** 2
    wy = 4.0 * math.sin(math.pi * (height // 2) / height) ** 2
    return math.sqrt(wx + wy)


def fidelity_value_closed(
    kind: str,
    x: torch.Tensor,
    y_obs: torch.Tensor | None,
) -> torch.Tensor:
    """Lower-semicontinuous closed fidelity used for primal/dual diagnostics."""
    k = kind.lower().strip()
    inf = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
    if k in ("gaussian", "l2"):
        if y_obs is None:
            raise ValueError("gaussian requires y_obs")
        return 0.5 * ((x - y_obs) ** 2).sum()
    if k == "poisson_intensity":
        if y_obs is None or bool((y_obs < 0).any().item()):
            raise ValueError("poisson_intensity requires y_obs >= 0")
        if bool((x < 0).any().item()) or bool(((x == 0) & (y_obs > 0)).any().item()):
            return inf
        positive_y = y_obs > 0
        log_x = torch.where(positive_y, torch.log(x), torch.zeros_like(x))
        return (x - torch.where(positive_y, y_obs * log_x, torch.zeros_like(x))).sum()
    if k == "poisson_log":
        if y_obs is None:
            raise ValueError("poisson_log requires y_obs")
        return (torch.exp(x) - y_obs * x).sum()
    if k == "kl":
        if y_obs is None or bool((y_obs <= 0).any().item()):
            raise ValueError("KL requires y_obs > 0")
        if bool((x < 0).any().item()):
            return inf
        positive_x = x > 0
        term = torch.where(
            positive_x,
            x * (torch.log(x) - torch.log(y_obs)) - x + y_obs,
            y_obs,
        )
        return term.sum()
    if k == "xlogx":
        if bool((x < 0).any().item()):
            return inf
        return torch.where(x > 0, x * torch.log(x), torch.zeros_like(x)).sum()
    if k == "exp":
        return torch.exp(x).sum()
    if k == "neglog":
        if bool((x <= 0).any().item()):
            return inf
        return (-torch.log(x)).sum()
    raise ValueError(f"unsupported fidelity kind: {kind!r}")


def fidelity_conjugate_value(
    kind: str,
    s: torch.Tensor,
    y_obs: torch.Tensor | None,
) -> torch.Tensor:
    """Value of F*(s) for the five well-posed PDHG fidelities."""
    k = kind.lower().strip()
    inf = torch.full((), float("inf"), dtype=s.dtype, device=s.device)
    if k in ("gaussian", "l2"):
        if y_obs is None:
            raise ValueError("gaussian requires y_obs")
        return (0.5 * s * s + s * y_obs).sum()
    if k == "poisson_intensity":
        if y_obs is None or bool((y_obs < 0).any().item()):
            raise ValueError("poisson_intensity requires y_obs >= 0")
        if bool((s > 1).any().item()) or bool(((s == 1) & (y_obs > 0)).any().item()):
            return inf
        positive_y = y_obs > 0
        term = torch.where(
            positive_y,
            y_obs * (torch.log(y_obs) - torch.log1p(-s) - 1.0),
            torch.zeros_like(s),
        )
        return term.sum()
    if k == "poisson_log":
        if y_obs is None:
            raise ValueError("poisson_log requires y_obs")
        t = s + y_obs
        if bool((t < 0).any().item()):
            return inf
        return torch.where(t > 0, t * torch.log(t) - t, torch.zeros_like(t)).sum()
    if k == "kl":
        if y_obs is None or bool((y_obs <= 0).any().item()):
            raise ValueError("KL requires y_obs > 0")
        return (y_obs * (torch.exp(s) - 1.0)).sum()
    if k == "xlogx":
        return torch.exp(s - 1.0).sum()
    raise ValueError(f"no finite-model conjugate implemented for {kind!r}")


def primal_objective(
    kind: str,
    x: torch.Tensor,
    y_obs: torch.Tensor | None,
    lam_tv: float,
) -> torch.Tensor:
    return fidelity_value_closed(kind, x, y_obs) + float(lam_tv) * legacy.tv_iso(x)


def diagnose_solution(
    kind: str,
    x: torch.Tensor,
    dual: torch.Tensor,
    y_obs: torch.Tensor | None,
    lam_tv: float,
    *,
    tau: float,
    sigma: float,
) -> Close1Diagnostics:
    """Compute gap and normalized resolvent residuals at one PDHG state."""
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be positive")
    with torch.no_grad():
        kx = legacy.K(x)
        ktp = legacy.Kt(dual)
        dual_norm = torch.sqrt(dual[:, :1] ** 2 + dual[:, 1:] ** 2)
        dual_violation_t = torch.clamp(dual_norm - float(lam_tv), min=0.0).max()

        x_fixed = legacy.canonical_prox_f(kind, x - float(tau) * ktp, float(tau), y_obs)
        dual_fixed = legacy.prox_gstar_tv_iso(
            dual + float(sigma) * kx, float(lam_tv)
        )
        rx = (x - x_fixed).norm() / float(tau)
        rp = (dual - dual_fixed).norm() / float(sigma)
        rx_scale = max(1.0, float(x.norm().item()), float(ktp.norm().item()))
        rp_scale = max(1.0, float(dual.norm().item()), float(kx.norm().item()))
        rx_n = float(rx.item()) / rx_scale
        rp_n = float(rp.item()) / rp_scale

        primal_t = primal_objective(kind, x, y_obs, lam_tv)
        fstar_t = fidelity_conjugate_value(kind, -ktp, y_obs)
        primal = float(primal_t.item())
        feasible = float(dual_violation_t.item()) <= 16.0 * torch.finfo(x.dtype).eps
        dual_obj = -float(fstar_t.item()) if feasible else float("-inf")
        gap = primal - dual_obj
        scale = max(1.0, abs(primal), abs(dual_obj)) if math.isfinite(dual_obj) else float("inf")
        rel_gap = abs(gap) / scale if math.isfinite(gap) and math.isfinite(scale) else float("inf")
        finite = all(math.isfinite(v) for v in (primal, dual_obj, gap, rel_gap, rx_n, rp_n))
        return Close1Diagnostics(
            primal_objective=primal,
            dual_objective=dual_obj,
            primal_dual_gap=gap,
            relative_gap=rel_gap,
            kkt_primal_residual=rx_n,
            kkt_dual_residual=rp_n,
            kkt_max_residual=max(rx_n, rp_n),
            dual_feasibility_violation=float(dual_violation_t.item()),
            finite=finite,
        )


def pdhg(
    kind: str,
    x0: torch.Tensor,
    y_obs: torch.Tensor | None,
    lam_tv: float,
    *,
    kkt_tol: float = 1.0e-5,
    gap_tol: float = 1.0e-5,
    norm_k: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, PDHGInfoClose1]:
    """Run the immutable promoted solver and attach CLOSE1 diagnostics."""
    if x0.ndim != 4:
        raise ValueError("x0 must be a 4-D tensor")
    if kkt_tol < 0 or gap_tol < 0 or not math.isfinite(kkt_tol + gap_tol):
        raise ValueError("kkt_tol and gap_tol must be finite and nonnegative")
    exact_norm = norm_k_periodic_2d(int(x0.shape[-2]), int(x0.shape[-1]))
    if norm_k is None:
        step_bound = exact_norm if exact_norm > 0.0 else norm_k_upper_bound_periodic_2d()
    else:
        step_bound = float(norm_k)
        slack = 64.0 * math.ulp(max(1.0, exact_norm))
        if step_bound + slack < exact_norm:
            raise ValueError(
                f"norm_k={step_bound} underestimates the exact finite-grid norm {exact_norm}"
            )

    x, dual, old = legacy.pdhg(
        kind, x0, y_obs, lam_tv, norm_k=step_bound, **kwargs
    )
    t0 = time.perf_counter()
    diagnostics = diagnose_solution(
        kind, x, dual, y_obs, lam_tv, tau=old.tau, sigma=old.sigma
    )
    diagnostics_elapsed = time.perf_counter() - t0
    certified = bool(
        diagnostics.finite
        and diagnostics.kkt_max_residual <= kkt_tol
        and diagnostics.relative_gap <= gap_tol
        and diagnostics.dual_feasibility_violation <= kkt_tol
    )
    stabilized = bool(old.converged)
    stop_reason = (
        "relative-change tolerance met" if stabilized else "maximum iterations reached"
    )
    history = {key: list(value) for key, value in old.history.items()}
    history.update({
        "final_relative_gap": [diagnostics.relative_gap],
        "final_kkt_primal_residual": [diagnostics.kkt_primal_residual],
        "final_kkt_dual_residual": [diagnostics.kkt_dual_residual],
    })
    info = PDHGInfoClose1(
        iterations=old.iterations,
        stabilized=stabilized,
        certified=certified,
        stop_reason=stop_reason,
        tau=old.tau,
        sigma=old.sigma,
        norm_k_exact=exact_norm,
        norm_k_step_bound=step_bound,
        backend=old.backend,
        primal_rel_change=old.primal_rel_change,
        dual_rel_change=old.dual_rel_change,
        diagnostics=diagnostics,
        solver_elapsed_seconds=old.elapsed_seconds,
        diagnostics_elapsed_seconds=diagnostics_elapsed,
        history=history,
        status_note=(
            "Numerical diagnostic thresholds passed; this is not a floating-point "
            "convergence proof." if certified else
            "End state is not certified at the requested KKT/gap tolerances."
        ),
    )
    return x, dual, info


__all__ = [
    "Close1Diagnostics", "PDHGInfoClose1",
    "norm_k_upper_bound_periodic_2d", "norm_k_periodic_2d",
    "fidelity_value_closed", "fidelity_conjugate_value", "primal_objective",
    "diagnose_solution", "pdhg",
]
