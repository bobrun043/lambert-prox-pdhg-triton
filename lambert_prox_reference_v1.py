"""Canonical FP64 reference for the Lambert-Prox project.

This module implements the normative bi-coordinate inverse

    U(R) = phi^{-1}(R),  phi(u) = u + log(u),  u > 0,

without forming exp(R) in regimes where that would overflow.  It is an
independent CPU/NumPy oracle: it does not import any Torch or Triton project
module.

Normative source:
    LAMBERT_PROX_SPEC_CANONIQUE_V1_2026-07-21.pdf

The central contract returns both:
    u : value coordinate, possibly underflowing to 0 in FP64;
    q : logarithmic coordinate log(u), which remains meaningful;
plus an explicit status and a residual evaluated in the active coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class SolveStatus(IntEnum):
    """Elementwise status codes for the canonical scalar inverse."""

    OK_VALUE = 0
    OK_LOG_ONLY = 1
    INVALID_INPUT = 2
    MAX_ITER = 3
    NUMERICAL_FAILURE = 4


@dataclass(frozen=True)
class ULogUResult:
    """Result of the bi-coordinate solve.

    All fields are NumPy arrays with the broadcast/input shape, including for
    scalar input (shape ``()``).
    """

    u: np.ndarray
    q: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    residual: np.ndarray
    step: np.ndarray

    @property
    def converged(self) -> np.ndarray:
        return (self.status == int(SolveStatus.OK_VALUE)) | (
            self.status == int(SolveStatus.OK_LOG_ONLY)
        )

    @property
    def log_only(self) -> np.ndarray:
        return self.status == int(SolveStatus.OK_LOG_ONLY)


def _as_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _tolerances(
    R: np.ndarray, coordinate: np.ndarray, atol: float, rtol: float
) -> tuple[np.ndarray, np.ndarray]:
    res_tol = atol + rtol * np.maximum(1.0, np.abs(R))
    step_tol = atol + rtol * np.maximum(1.0, np.abs(coordinate))
    return res_tol, step_tol


def solve_u_log_u_reference(
    R: Any,
    *,
    max_iter: int = 20,
    atol: float | None = None,
    rtol: float | None = None,
    r_switch: float = -8.0,
) -> ULogUResult:
    """Solve ``u + log(u) = R`` in FP64 using two coordinates.

    Parameters
    ----------
    R:
        Finite real scalar or array.
    max_iter:
        Maximum adaptive Halley iterations. Must be positive.
    atol, rtol:
        Double stopping criterion on residual and step. Defaults are
        ``8 * eps(float64)``.
    r_switch:
        Values below this threshold are solved in q=log(u) coordinates.
        The default -8 follows the canonical specification.

    Returns
    -------
    ULogUResult
        Explicit value/log coordinates, status, iteration count, residual,
        and final step. Residual is ``|q + exp(q) - R|`` in log-coordinate
        mode and ``|u + log(u) - R|`` in value-coordinate mode.

    Notes
    -----
    * No input clamp changes the mathematical problem.
    * Underflow of ``u = exp(q)`` is reported as ``OK_LOG_ONLY`` rather than
      as a solver failure.
    * The value-coordinate Halley step is evaluated in a scaled algebraically
      equivalent form to avoid overflow of ``(u+1)^2`` for huge R.
    """

    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(r_switch):
        raise ValueError("r_switch must be finite")

    eps = np.finfo(np.float64).eps
    atol = 8.0 * eps if atol is None else float(atol)
    rtol = 8.0 * eps if rtol is None else float(rtol)
    if not (np.isfinite(atol) and atol >= 0.0):
        raise ValueError("atol must be finite and non-negative")
    if not (np.isfinite(rtol) and rtol >= 0.0):
        raise ValueError("rtol must be finite and non-negative")

    Rv = _as_f64(R)
    shape = Rv.shape

    u_out = np.full(shape, np.nan, dtype=np.float64)
    q_out = np.full(shape, np.nan, dtype=np.float64)
    status = np.full(shape, int(SolveStatus.INVALID_INPUT), dtype=np.int8)
    iterations = np.zeros(shape, dtype=np.int16)
    residual = np.full(shape, np.nan, dtype=np.float64)
    step_out = np.full(shape, np.nan, dtype=np.float64)

    valid = np.isfinite(Rv)
    if not np.any(valid):
        return ULogUResult(u_out, q_out, status, iterations, residual, step_out)

    # ---- Log coordinate: h(q) = q + exp(q) - R ----
    log_mask = valid & (Rv < r_switch)
    if np.any(log_mask):
        idx = np.flatnonzero(log_mask.ravel())
        Rloc = Rv.ravel()[idx]
        q = Rloc.copy()  # canonical q0 = R
        active = np.ones(q.shape, dtype=bool)
        itloc = np.zeros(q.shape, dtype=np.int16)
        final_step = np.full(q.shape, np.nan, dtype=np.float64)
        failed = np.zeros(q.shape, dtype=bool)

        for k in range(1, int(max_iter) + 1):
            if not np.any(active):
                break
            ia = np.flatnonzero(active)
            qa = q[ia]
            Ra = Rloc[ia]
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                e = np.exp(qa)
                f = qa + e - Ra
                fp = 1.0 + e
                den = 2.0 * fp * fp - f * e
                delta = (2.0 * f * fp) / den
                q_new = qa - delta
                e_new = np.exp(q_new)
                f_new = q_new + e_new - Ra

            bad = (~np.isfinite(q_new)) | (~np.isfinite(f_new)) | (~np.isfinite(delta)) | (den == 0.0)
            if np.any(bad):
                bad_global = ia[bad]
                failed[bad_global] = True
                active[bad_global] = False
                itloc[bad_global] = k

            good_local = ~bad
            if np.any(good_local):
                good_global = ia[good_local]
                q[good_global] = q_new[good_local]
                final_step[good_global] = delta[good_local]
                res_tol, step_tol = _tolerances(
                    Ra[good_local], q_new[good_local], atol, rtol
                )
                done = (np.abs(f_new[good_local]) <= res_tol) & (
                    np.abs(delta[good_local]) <= step_tol
                )
                if np.any(done):
                    done_global = good_global[done]
                    active[done_global] = False
                    itloc[done_global] = k

        maxed = active & ~failed
        itloc[maxed] = int(max_iter)

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            uloc = np.exp(q)
            rloc = np.abs(q + np.exp(q) - Rloc)

        stloc = np.full(q.shape, int(SolveStatus.OK_VALUE), dtype=np.int8)
        stloc[(uloc == 0.0) & np.isfinite(q) & ~maxed & ~failed] = int(
            SolveStatus.OK_LOG_ONLY
        )
        stloc[maxed] = int(SolveStatus.MAX_ITER)
        stloc[failed] = int(SolveStatus.NUMERICAL_FAILURE)

        u_out.ravel()[idx] = uloc
        q_out.ravel()[idx] = q
        status.ravel()[idx] = stloc
        iterations.ravel()[idx] = itloc
        residual.ravel()[idx] = rloc
        step_out.ravel()[idx] = final_step

    # ---- Value coordinate: g(u) = u + log(u) - R ----
    val_mask = valid & ~log_mask
    if np.any(val_mask):
        idx = np.flatnonzero(val_mask.ravel())
        Rloc = Rv.ravel()[idx]
        u = np.empty_like(Rloc)

        lo = Rloc < -0.3
        mid = (Rloc >= -0.3) & (Rloc <= 8.0)
        hi = Rloc > 8.0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            u[lo] = np.exp(Rloc[lo])
            u[mid] = 1.0 + Rloc[mid]
            L = np.log(Rloc[hi])
            u[hi] = Rloc[hi] - L + L / Rloc[hi]

        active = np.ones(u.shape, dtype=bool)
        itloc = np.zeros(u.shape, dtype=np.int16)
        final_step = np.full(u.shape, np.nan, dtype=np.float64)
        failed = (~np.isfinite(u)) | (u <= 0.0)
        active[failed] = False
        itloc[failed] = 0

        for k in range(1, int(max_iter) + 1):
            if not np.any(active):
                break
            ia = np.flatnonzero(active)
            ua = u[ia]
            Ra = Rloc[ia]
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                f = ua + np.log(ua) - Ra
                inv = 1.0 / ua
                one = 1.0 + inv
                # Algebraically equal to f*u*(u+1)/((u+1)^2 + 0.5*f),
                # scaled by u^2 to remain finite for huge u.
                den = one * one + 0.5 * f * inv * inv
                delta = f * one / den
                u_new = ua - delta
                f_new = u_new + np.log(u_new) - Ra

            bad = (
                (~np.isfinite(u_new))
                | (u_new <= 0.0)
                | (~np.isfinite(f_new))
                | (~np.isfinite(delta))
                | (den == 0.0)
            )
            if np.any(bad):
                bad_global = ia[bad]
                failed[bad_global] = True
                active[bad_global] = False
                itloc[bad_global] = k

            good_local = ~bad
            if np.any(good_local):
                good_global = ia[good_local]
                u[good_global] = u_new[good_local]
                final_step[good_global] = delta[good_local]
                res_tol, step_tol = _tolerances(
                    Ra[good_local], u_new[good_local], atol, rtol
                )
                done = (np.abs(f_new[good_local]) <= res_tol) & (
                    np.abs(delta[good_local]) <= step_tol
                )
                if np.any(done):
                    done_global = good_global[done]
                    active[done_global] = False
                    itloc[done_global] = k

        maxed = active & ~failed
        itloc[maxed] = int(max_iter)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            q = np.log(u)
            rloc = np.abs(u + q - Rloc)

        stloc = np.full(u.shape, int(SolveStatus.OK_VALUE), dtype=np.int8)
        stloc[maxed] = int(SolveStatus.MAX_ITER)
        stloc[failed] = int(SolveStatus.NUMERICAL_FAILURE)

        u_out.ravel()[idx] = u
        q_out.ravel()[idx] = q
        status.ravel()[idx] = stloc
        iterations.ravel()[idx] = itloc
        residual.ravel()[idx] = rloc
        step_out.ravel()[idx] = final_step

    return ULogUResult(u_out, q_out, status, iterations, residual, step_out)


def solve_u_log_u(R: Any, **kwargs: Any) -> np.ndarray:
    """Convenience value-only wrapper.

    Use :func:`solve_u_log_u_reference` whenever underflow status or q is
    relevant. This wrapper deliberately does not hide that distinction in its
    documentation.
    """

    return solve_u_log_u_reference(R, **kwargs).u


def derivative_u_from_result(result: ULogUResult) -> np.ndarray:
    """Exact first derivative dU/dR = u/(1+u) in value coordinates."""

    with np.errstate(invalid="ignore", divide="ignore"):
        return result.u / (1.0 + result.u)


def derivative_q_from_result(result: ULogUResult) -> np.ndarray:
    """Exact first derivative dq/dR = 1/(1+u), stable in log-only regimes."""

    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 / (1.0 + result.u)


# ---------------------------------------------------------------------------
# Canonical FP64 proximal wrappers. They are reference formulas, not a claim
# of novelty. All domain checks are mathematical checks, never silent clamps.
# ---------------------------------------------------------------------------


def _broadcast_f64(*xs: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.broadcast_arrays(*[_as_f64(x) for x in xs]))


def _require_all(condition: np.ndarray, message: str) -> None:
    if not bool(np.all(condition)):
        raise ValueError(message)


def prox_exp(v: Any, lam: Any) -> np.ndarray:
    v, lam = _broadcast_f64(v, lam)
    _require_all(np.isfinite(v) & np.isfinite(lam) & (lam > 0.0), "prox_exp requires finite v and lam > 0")
    result = solve_u_log_u_reference(v + np.log(lam))
    _require_all(result.converged, "canonical primitive failed in prox_exp")
    # Equivalent reconstruction x = q - log(lam) avoids cancellation
    # when v and u are both large.
    return result.q - np.log(lam)


def prox_xlogx(v: Any, lam: Any) -> np.ndarray:
    v, lam = _broadcast_f64(v, lam)
    _require_all(np.isfinite(v) & np.isfinite(lam) & (lam > 0.0), "prox_xlogx requires finite v and lam > 0")
    result = solve_u_log_u_reference(v / lam - 1.0 - np.log(lam))
    _require_all(result.converged, "canonical primitive failed in prox_xlogx")
    return lam * result.u


def prox_kl(v: Any, y: Any, lam: Any) -> np.ndarray:
    v, y, lam = _broadcast_f64(v, y, lam)
    _require_all(np.isfinite(v) & np.isfinite(y) & np.isfinite(lam), "prox_kl requires finite inputs")
    _require_all(y >= 0.0, "prox_kl requires y >= 0")
    _require_all(lam > 0.0, "prox_kl requires lam > 0")
    out = np.zeros_like(v)
    positive = y > 0.0
    if np.any(positive):
        result = solve_u_log_u_reference(
            v[positive] / lam[positive] + np.log(y[positive]) - np.log(lam[positive])
        )
        _require_all(result.converged, "canonical primitive failed in prox_kl")
        out[positive] = lam[positive] * result.u
    # y == 0 is the exact model boundary: prox = 0.
    return out


def prox_poisson_log(v: Any, y: Any, lam: Any) -> np.ndarray:
    v, y, lam = _broadcast_f64(v, y, lam)
    _require_all(np.isfinite(v) & np.isfinite(y) & np.isfinite(lam), "prox_poisson_log requires finite inputs")
    _require_all(y >= 0.0, "prox_poisson_log requires y >= 0")
    _require_all(lam > 0.0, "prox_poisson_log requires lam > 0")
    d = v + lam * y
    result = solve_u_log_u_reference(d + np.log(lam))
    _require_all(result.converged, "canonical primitive failed in prox_poisson_log")
    # Since d + log(lam) = u + q, x = d-u = q-log(lam).
    # The q-form avoids cancellation for large d and u.
    return result.q - np.log(lam)


def prox_poisson_intensity(v: Any, y: Any, lam: Any) -> np.ndarray:
    v, y, lam = _broadcast_f64(v, y, lam)
    _require_all(np.isfinite(v) & np.isfinite(y) & np.isfinite(lam), "prox_poisson_intensity requires finite inputs")
    _require_all(y >= 0.0, "prox_poisson_intensity requires y >= 0")
    _require_all(lam > 0.0, "prox_poisson_intensity requires lam > 0")
    a = v - lam
    s = np.sqrt(a * a + 4.0 * lam * y)
    out = np.empty_like(a)
    nonnegative = a >= 0.0
    out[nonnegative] = 0.5 * (a[nonnegative] + s[nonnegative])
    negative = ~nonnegative
    denom = s[negative] - a[negative]
    out[negative] = np.divide(
        2.0 * lam[negative] * y[negative],
        denom,
        out=np.zeros_like(denom),
        where=denom != 0.0,
    )
    return out


def prox_neglog(v: Any, lam: Any) -> np.ndarray:
    v, lam = _broadcast_f64(v, lam)
    _require_all(np.isfinite(v) & np.isfinite(lam), "prox_neglog requires finite inputs")
    _require_all(lam > 0.0, "prox_neglog requires lam > 0")
    s = np.sqrt(v * v + 4.0 * lam)
    out = np.empty_like(v)
    nonnegative = v >= 0.0
    out[nonnegative] = 0.5 * (v[nonnegative] + s[nonnegative])
    negative = ~nonnegative
    out[negative] = 2.0 * lam[negative] / (s[negative] - v[negative])
    return out


def prox_gaussian(v: Any, y: Any, lam: Any) -> np.ndarray:
    v, y, lam = _broadcast_f64(v, y, lam)
    _require_all(np.isfinite(v) & np.isfinite(y) & np.isfinite(lam), "prox_gaussian requires finite inputs")
    _require_all(lam > 0.0, "prox_gaussian requires lam > 0")
    return (v + lam * y) / (1.0 + lam)


__all__ = [
    "SolveStatus",
    "ULogUResult",
    "solve_u_log_u_reference",
    "solve_u_log_u",
    "derivative_u_from_result",
    "derivative_q_from_result",
    "prox_exp",
    "prox_xlogx",
    "prox_kl",
    "prox_poisson_log",
    "prox_poisson_intensity",
    "prox_neglog",
    "prox_gaussian",
]
