"""Canonical PyTorch backend for the Lambert-Prox project.

This module is the unique Torch implementation corresponding to the FP64
NumPy oracle ``lambert_prox_reference_v1.py``.  It intentionally does not
import any earlier project implementation.

Mathematical primitive
----------------------
Solve, elementwise,

    u + log(u) = R,  u > 0,

while retaining both coordinates

    u       (value coordinate),
    q=log u (logarithmic coordinate).

The logarithmic coordinate remains meaningful when ``u`` underflows.  The
implementation uses ordinary differentiable Torch operations and fixed Halley
passes, so first and second derivatives are obtained by native autograd rather
than by an opaque custom backward.

Scope of V1
-----------
* Supported input dtypes: torch.float32 and torch.float64.
* Device agnostic: CPU and CUDA use the same formulas; this release was
  executed and closed on CPU only unless a CUDA report is explicitly present.
* No performance or originality claim is made by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch


class SolveStatus(IntEnum):
    """Elementwise status codes, aligned with the NumPy reference."""

    OK_VALUE = 0
    OK_LOG_ONLY = 1
    INVALID_INPUT = 2
    MAX_ITER = 3
    NUMERICAL_FAILURE = 4


@dataclass(frozen=True)
class TorchULogUResult:
    """Bi-coordinate result of ``u + log(u) = R``."""

    u: torch.Tensor
    q: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    residual: torch.Tensor
    step: torch.Tensor

    @property
    def converged(self) -> torch.Tensor:
        return (self.status == int(SolveStatus.OK_VALUE)) | (
            self.status == int(SolveStatus.OK_LOG_ONLY)
        )

    @property
    def log_only(self) -> torch.Tensor:
        return self.status == int(SolveStatus.OK_LOG_ONLY)


def _require_supported_float(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name} must have dtype torch.float32 or torch.float64; got {x.dtype}"
        )


def _require_all(condition: torch.Tensor, message: str) -> None:
    # Validation is intentionally explicit.  On CUDA this synchronizes; callers
    # that need an asynchronous production path may set validate_args=False only
    # after validating inputs at a higher level.
    if not bool(torch.all(condition).item()):
        raise ValueError(message)


def _as_like(x: Any, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _broadcast_like(first: torch.Tensor, *others: Any) -> tuple[torch.Tensor, ...]:
    _require_supported_float(first, "first input")
    tensors = [first]
    tensors.extend(_as_like(x, first) for x in others)
    return tuple(torch.broadcast_tensors(*tensors))


def _default_tol(dtype: torch.dtype) -> tuple[float, float]:
    eps = torch.finfo(dtype).eps
    # Fixed-iteration Torch path: a slightly wider status threshold than the
    # adaptive FP64 oracle, while remaining at the dtype rounding scale.
    factor = 32.0
    return factor * eps, factor * eps


def solve_u_log_u_pair(
    R: torch.Tensor,
    *,
    iters: int = 6,
    r_switch: float = -8.0,
    atol: float | None = None,
    rtol: float | None = None,
) -> TorchULogUResult:
    """Solve ``u + log(u) = R`` with value and log coordinates.

    The iteration count is fixed to make the numerical path deterministic and
    compiler/GPU friendly. Invalid inputs are reported elementwise rather than
    silently clamped.
    """

    _require_supported_float(R, "R")
    if not isinstance(iters, int) or iters <= 0:
        raise ValueError("iters must be a positive integer")
    if not torch.isfinite(torch.tensor(r_switch, dtype=R.dtype, device=R.device)):
        raise ValueError("r_switch must be finite")

    atol_default, rtol_default = _default_tol(R.dtype)
    atol_v = atol_default if atol is None else float(atol)
    rtol_v = rtol_default if rtol is None else float(rtol)
    if not (atol_v >= 0.0 and rtol_v >= 0.0):
        raise ValueError("atol and rtol must be non-negative")

    finite = torch.isfinite(R)
    zero = torch.zeros((), dtype=R.dtype, device=R.device)
    Rw = torch.where(finite, R, zero)
    log_mask = finite & (Rw < r_switch)
    val_mask = finite & ~log_mask

    # ---- Log coordinate: h(q)=q+exp(q)-R ----
    # The entire inactive branch is replaced by a benign constant problem.
    # This is necessary because torch.where may still propagate NaNs from an
    # unselected branch during higher-order differentiation.
    Rq = torch.where(log_mask, Rw, torch.full_like(Rw, float(r_switch)))
    q = Rq.clone()
    q_step = torch.zeros_like(q)
    for _ in range(iters):
        e = torch.exp(q)
        f = q + e - Rq
        fp = 1.0 + e
        den = 2.0 * fp * fp - f * e
        q_step = (2.0 * f * fp) / den
        q = q - q_step

    # ---- Value coordinate: g(u)=u+log(u)-R ----
    # As above, inactive elements solve the benign problem R=0.
    Ru = torch.where(val_mask, Rw, torch.zeros_like(Rw))
    lo = Ru < -0.3
    hi = Ru > 8.0
    # Safe eager evaluation on inactive branches.
    u_lo = torch.exp(torch.minimum(Ru, torch.full_like(Ru, -0.3)))
    R_for_log = torch.maximum(Ru, torch.full_like(Ru, 8.0))
    log_R = torch.log(R_for_log)
    u_hi = Ru - log_R + log_R / R_for_log
    u_mid = 1.0 + Ru
    u = torch.where(lo, u_lo, torch.where(hi, u_hi, u_mid))

    u_step = torch.zeros_like(u)
    for _ in range(iters):
        f = u + torch.log(u) - Ru
        inv = 1.0 / u
        one = 1.0 + inv
        den = one * one + 0.5 * f * inv * inv
        u_step = f * one / den
        u = u - u_step

    q_from_u = torch.log(u)
    u_from_q = torch.exp(q)

    out_u = torch.where(log_mask, u_from_q, u)
    out_q = torch.where(log_mask, q, q_from_u)
    out_step = torch.where(log_mask, q_step, u_step)
    residual_log = torch.abs(q + torch.exp(q) - Rq)
    residual_value = torch.abs(u + torch.log(u) - Ru)
    residual = torch.where(log_mask, residual_log, residual_value)

    numerical_ok = (
        torch.isfinite(out_q)
        & torch.isfinite(residual)
        & torch.isfinite(out_step)
        & (out_u >= 0.0)
    )
    res_tol = atol_v + rtol_v * torch.maximum(torch.ones_like(Rw), torch.abs(Rw))
    coord = torch.where(log_mask, out_q, out_u)
    step_tol = atol_v + rtol_v * torch.maximum(torch.ones_like(coord), torch.abs(coord))
    converged = finite & numerical_ok & (residual <= res_tol) & (torch.abs(out_step) <= step_tol)

    status = torch.full(R.shape, int(SolveStatus.INVALID_INPUT), dtype=torch.int8, device=R.device)
    status = torch.where(
        finite & ~numerical_ok,
        torch.full_like(status, int(SolveStatus.NUMERICAL_FAILURE)),
        status,
    )
    status = torch.where(
        finite & numerical_ok & ~converged,
        torch.full_like(status, int(SolveStatus.MAX_ITER)),
        status,
    )
    status = torch.where(
        converged,
        torch.full_like(status, int(SolveStatus.OK_VALUE)),
        status,
    )
    log_only = converged & (out_u == 0.0) & torch.isfinite(out_q)
    status = torch.where(
        log_only,
        torch.full_like(status, int(SolveStatus.OK_LOG_ONLY)),
        status,
    )

    iterations = torch.where(
        finite,
        torch.full(R.shape, int(iters), dtype=torch.int16, device=R.device),
        torch.zeros(R.shape, dtype=torch.int16, device=R.device),
    )
    nan = torch.full_like(R, float("nan"))
    out_u = torch.where(finite, out_u, nan)
    out_q = torch.where(finite, out_q, nan)
    residual = torch.where(finite, residual, nan)
    out_step = torch.where(finite, out_step, nan)

    return TorchULogUResult(out_u, out_q, status, iterations, residual, out_step)


def solve_u_log_u(R: torch.Tensor, *, iters: int = 6) -> torch.Tensor:
    """Value-only convenience wrapper. Use the pair API for extreme negatives."""

    return solve_u_log_u_pair(R, iters=iters).u


def derivative_u(result: TorchULogUResult) -> torch.Tensor:
    """Exact implicit derivative ``du/dR = u/(1+u)``."""

    return result.u / (1.0 + result.u)


def derivative_q(result: TorchULogUResult) -> torch.Tensor:
    """Exact implicit derivative ``dq/dR = 1/(1+u)``."""

    return 1.0 / (1.0 + result.u)


def _validate_finite(*xs: torch.Tensor, message: str) -> None:
    cond = torch.ones_like(xs[0], dtype=torch.bool)
    for x in xs:
        cond = cond & torch.isfinite(x)
    _require_all(cond, message)


def prox_exp(v: torch.Tensor, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    v, lam = _broadcast_like(v, lam)
    if validate_args:
        _validate_finite(v, lam, message="prox_exp requires finite v and lam")
        _require_all(lam > 0.0, "prox_exp requires lam > 0")
    result = solve_u_log_u_pair(v + torch.log(lam), iters=iters)
    if validate_args:
        _require_all(result.converged, "canonical primitive failed in prox_exp")
    return result.q - torch.log(lam)


def _prox_xlogx_components(
    v: torch.Tensor, lam: Any, *, iters: int, validate_args: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    v, lam = _broadcast_like(v, lam)
    if validate_args:
        _validate_finite(v, lam, message="prox_xlogx requires finite v and lam")
        _require_all(lam > 0.0, "prox_xlogx requires lam > 0")
    log_lam = torch.log(lam)
    result = solve_u_log_u_pair(v / lam - 1.0 - log_lam, iters=iters)
    if validate_args:
        _require_all(result.converged, "canonical primitive failed in prox_xlogx")
    return lam * result.u, log_lam + result.q


def prox_xlogx(v: torch.Tensor, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    """Value coordinate of the xlogx prox; may underflow to zero."""
    return _prox_xlogx_components(v, lam, iters=iters, validate_args=validate_args)[0]


def prox_xlogx_log(v: torch.Tensor, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    """Stable ``log(prox_xlogx(v, lam))`` including value-underflow regimes."""
    return _prox_xlogx_components(v, lam, iters=iters, validate_args=validate_args)[1]


def _prox_kl_components(
    v: torch.Tensor, y: Any, lam: Any, *, iters: int, validate_args: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    v, y, lam = _broadcast_like(v, y, lam)
    if validate_args:
        _validate_finite(v, y, lam, message="prox_kl requires finite inputs")
        _require_all(y >= 0.0, "prox_kl requires y >= 0")
        _require_all(lam > 0.0, "prox_kl requires lam > 0")
    positive = y > 0.0
    y_for_log = torch.where(positive, y, torch.ones_like(y))
    R = v / lam + torch.log(y_for_log) - torch.log(lam)
    result = solve_u_log_u_pair(R, iters=iters)
    if validate_args:
        _require_all((~positive) | result.converged, "canonical primitive failed in prox_kl")
    positive_out = lam * result.u
    positive_log = torch.log(lam) + result.q
    value = torch.where(positive, positive_out, torch.zeros_like(positive_out))
    log_value = torch.where(positive, positive_log, torch.full_like(positive_log, -float("inf")))
    return value, log_value


def prox_kl(v: torch.Tensor, y: Any, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    """Value coordinate of the KL prox; y=0 is the exact zero boundary."""
    return _prox_kl_components(v, y, lam, iters=iters, validate_args=validate_args)[0]


def prox_kl_log(v: torch.Tensor, y: Any, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    """Stable log-coordinate of the KL prox; returns -inf at the exact y=0 boundary."""
    return _prox_kl_components(v, y, lam, iters=iters, validate_args=validate_args)[1]


def prox_poisson_log(v: torch.Tensor, y: Any, lam: Any, *, iters: int = 6, validate_args: bool = True) -> torch.Tensor:
    v, y, lam = _broadcast_like(v, y, lam)
    if validate_args:
        _validate_finite(v, y, lam, message="prox_poisson_log requires finite inputs")
        _require_all(y >= 0.0, "prox_poisson_log requires y >= 0")
        _require_all(lam > 0.0, "prox_poisson_log requires lam > 0")
    d = v + lam * y
    result = solve_u_log_u_pair(d + torch.log(lam), iters=iters)
    if validate_args:
        _require_all(result.converged, "canonical primitive failed in prox_poisson_log")
    return result.q - torch.log(lam)


def prox_poisson_intensity(v: torch.Tensor, y: Any, lam: Any, *, validate_args: bool = True) -> torch.Tensor:
    v, y, lam = _broadcast_like(v, y, lam)
    if validate_args:
        _validate_finite(v, y, lam, message="prox_poisson_intensity requires finite inputs")
        _require_all(y >= 0.0, "prox_poisson_intensity requires y >= 0")
        _require_all(lam > 0.0, "prox_poisson_intensity requires lam > 0")
    a = v - lam
    s = torch.hypot(a, 2.0 * torch.sqrt(lam * y))
    direct = 0.5 * (a + s)
    denom = s - a
    conjugate = torch.where(denom != 0.0, 2.0 * lam * y / denom, torch.zeros_like(denom))
    return torch.where(a >= 0.0, direct, conjugate)


def prox_neglog(v: torch.Tensor, lam: Any, *, validate_args: bool = True) -> torch.Tensor:
    v, lam = _broadcast_like(v, lam)
    if validate_args:
        _validate_finite(v, lam, message="prox_neglog requires finite inputs")
        _require_all(lam > 0.0, "prox_neglog requires lam > 0")
    s = torch.hypot(v, 2.0 * torch.sqrt(lam))
    direct = 0.5 * (v + s)
    conjugate = 2.0 * lam / (s - v)
    return torch.where(v >= 0.0, direct, conjugate)


def prox_gaussian(v: torch.Tensor, y: Any, lam: Any, *, validate_args: bool = True) -> torch.Tensor:
    v, y, lam = _broadcast_like(v, y, lam)
    if validate_args:
        _validate_finite(v, y, lam, message="prox_gaussian requires finite inputs")
        _require_all(lam > 0.0, "prox_gaussian requires lam > 0")
    return (v + lam * y) / (1.0 + lam)


__all__ = [
    "SolveStatus",
    "TorchULogUResult",
    "solve_u_log_u_pair",
    "solve_u_log_u",
    "derivative_u",
    "derivative_q",
    "prox_exp",
    "prox_xlogx",
    "prox_xlogx_log",
    "prox_kl",
    "prox_kl_log",
    "prox_poisson_log",
    "prox_poisson_intensity",
    "prox_neglog",
    "prox_gaussian",
]
