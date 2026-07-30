"""Canonical forward-only Triton elementwise prox candidates.

This module mirrors ``lambert_prox_torch_v1`` for CUDA float32 tensors.  It is
an acceleration candidate, not an independent mathematical implementation.
The public wrappers intentionally provide forward evaluation only; no autograd
claim is made until a separate backward audit is completed.

Canonical corrections relative to the legacy candidate
-------------------------------------------------------
* six fixed Halley passes and the same value/log coordinate split as Torch;
* exp and Poisson-log reconstructed as ``q - log(lambda)``;
* exact KL boundary ``y == 0 -> 0``;
* stable conjugate quadratic formulas for Poisson-intensity and -log.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import triton
import triton.language as tl

__all__ = [
    "prox_gaussian_axpy",
    "prox_poisson_intensity_axpy",
    "prox_poisson_log_axpy",
    "prox_poisson_nll_axpy",
    "prox_kl_axpy",
    "prox_xlogx_axpy",
    "prox_exp_axpy",
    "prox_neglog_axpy",
]

MODE_GAUSSIAN = 0
MODE_POISSON_INTENSITY = 1
MODE_POISSON_LOG = 2
MODE_KL = 3
MODE_XLOGX = 4
MODE_EXP = 5
MODE_NEGLOG = 6


def _as4(shape: Iterable[int]) -> tuple[int, int, int, int]:
    shape = tuple(shape)
    if len(shape) > 4:
        raise ValueError("canonical elementwise candidate supports at most 4 dimensions")
    return (1,) * (4 - len(shape)) + shape


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, int, int, int]:
    normalized = [_as4(s) for s in shapes]
    out: list[int] = []
    for dims in zip(*normalized):
        m = max(dims)
        if any(d not in (1, m) for d in dims):
            raise ValueError(f"non-broadcastable shapes: {shapes}")
        out.append(m)
    return tuple(out)  # type: ignore[return-value]


def _strides4(t: torch.Tensor, out_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    shape = _as4(tuple(t.shape))
    stride = (0,) * (4 - t.ndim) + tuple(t.stride())
    return tuple(stride[i] if shape[i] == out_shape[i] else 0 for i in range(4))  # type: ignore[return-value]


def _check_tensor(t: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype != torch.float32:
        raise TypeError(f"{name} must be float32; got {t.dtype}")
    if t.ndim > 4:
        raise ValueError(f"{name} must have at most 4 dimensions")


def _prepare(x: torch.Tensor, y: torch.Tensor, y_obs: torch.Tensor | None):
    _check_tensor(x, "x")
    _check_tensor(y, "y")
    tensors = [x, y]
    if y_obs is not None:
        _check_tensor(y_obs, "y_obs")
        tensors.append(y_obs)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    out_shape_native = torch.broadcast_shapes(*(tuple(t.shape) for t in tensors))
    out = torch.empty(out_shape_native, dtype=torch.float32, device=x.device)
    placeholder = x if y_obs is None else y_obs
    return out, placeholder, _as4(out_shape_native)


@triton.jit
def _hypot_pair(a, b):
    aa = tl.abs(a)
    bb = tl.abs(b)
    hi = tl.maximum(aa, bb)
    lo = tl.minimum(aa, bb)
    safe_hi = tl.maximum(hi, 1.0e-30)
    r = lo / safe_hi
    h = hi * tl.sqrt(1.0 + r * r)
    return tl.where(hi > 0.0, h, 0.0)


@triton.jit
def _solve_u_log_u_pair(R, ITERS: tl.constexpr = 6):
    """Canonical fixed-pass bi-coordinate primitive for float32."""
    log_mask = R < -8.0

    # Log coordinate h(q)=q+exp(q)-R. Inactive lanes solve R=-8.
    Rq = tl.where(log_mask, R, -8.0)
    q = Rq
    for _ in range(ITERS):
        e = tl.exp(q)
        f = q + e - Rq
        fp = 1.0 + e
        den = 2.0 * fp * fp - f * e
        q = q - (2.0 * f * fp) / den

    # Value coordinate g(u)=u+log(u)-R. Inactive lanes solve R=0.
    Ru = tl.where(log_mask, 0.0, R)
    lo = Ru < -0.3
    hi = Ru > 8.0
    u_lo = tl.exp(tl.minimum(Ru, -0.3))
    R_for_log = tl.maximum(Ru, 8.0)
    log_R = tl.log(R_for_log)
    u_hi = Ru - log_R + log_R / R_for_log
    u_mid = 1.0 + Ru
    u = tl.where(lo, u_lo, tl.where(hi, u_hi, u_mid))
    for _ in range(ITERS):
        f = u + tl.log(u) - Ru
        inv = 1.0 / u
        one = 1.0 + inv
        den = one * one + 0.5 * f * inv * inv
        u = u - f * one / den

    u_out = tl.where(log_mask, tl.exp(q), u)
    q_out = tl.where(log_mask, q, tl.log(u))
    return u_out, q_out


@triton.jit
def _prox_axpy_kernel(
    X, SX0: tl.int64, SX1: tl.int64, SX2: tl.int64, SX3: tl.int64,
    Y, SY0: tl.int64, SY1: tl.int64, SY2: tl.int64, SY3: tl.int64,
    YOBS, SYO0: tl.int64, SYO1: tl.int64, SYO2: tl.int64, SYO3: tl.int64,
    O,
    N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr,
    alpha: tl.float32, beta: tl.float32, gamma: tl.float32,
    lam: tl.float32,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    numel = N0 * N1 * N2 * N3
    mask = off < numel

    S0 = N1 * N2 * N3
    S1 = N2 * N3
    S2 = N3
    i0 = off // S0
    rem = off - i0 * S0
    i1 = rem // S1
    rem = rem - i1 * S1
    i2 = rem // S2
    i3 = rem - i2 * S2

    offx = i0 * SX0 + i1 * SX1 + i2 * SX2 + i3 * SX3
    offy = i0 * SY0 + i1 * SY1 + i2 * SY2 + i3 * SY3
    offo = off

    x = tl.load(X + offx, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offy, mask=mask, other=0.0).to(tl.float32)
    v = alpha * x + beta * y + gamma
    log_lam = tl.log(lam)

    if MODE == 0:
        offobs = i0 * SYO0 + i1 * SYO1 + i2 * SYO2 + i3 * SYO3
        obs = tl.load(YOBS + offobs, mask=mask, other=0.0).to(tl.float32)
        out = (v + lam * obs) / (1.0 + lam)

    elif MODE == 1:
        offobs = i0 * SYO0 + i1 * SYO1 + i2 * SYO2 + i3 * SYO3
        obs = tl.maximum(tl.load(YOBS + offobs, mask=mask, other=0.0).to(tl.float32), 0.0)
        a = v - lam
        b = 2.0 * tl.sqrt(lam * obs)
        s = _hypot_pair(a, b)
        direct = 0.5 * (a + s)
        denom = tl.maximum(s - a, 1.0e-30)
        conjugate = 2.0 * lam * obs / denom
        out = tl.where(a >= 0.0, direct, conjugate)

    elif MODE == 2:
        offobs = i0 * SYO0 + i1 * SYO1 + i2 * SYO2 + i3 * SYO3
        obs = tl.maximum(tl.load(YOBS + offobs, mask=mask, other=0.0).to(tl.float32), 0.0)
        d = v + lam * obs
        _, q = _solve_u_log_u_pair(d + log_lam)
        out = q - log_lam

    elif MODE == 3:
        offobs = i0 * SYO0 + i1 * SYO1 + i2 * SYO2 + i3 * SYO3
        obs = tl.maximum(tl.load(YOBS + offobs, mask=mask, other=0.0).to(tl.float32), 0.0)
        positive = obs > 0.0
        safe_obs = tl.where(positive, obs, 1.0)
        u, _ = _solve_u_log_u_pair(v / lam + tl.log(safe_obs) - log_lam)
        out = tl.where(positive, lam * u, 0.0)

    elif MODE == 4:
        u, _ = _solve_u_log_u_pair(v / lam - 1.0 - log_lam)
        out = lam * u

    elif MODE == 5:
        _, q = _solve_u_log_u_pair(v + log_lam)
        out = q - log_lam

    else:  # MODE == 6 (NEGLOG)
        b = 2.0 * tl.sqrt(lam)
        s = _hypot_pair(v, b)
        direct = 0.5 * (v + s)
        conjugate = 2.0 * lam / tl.maximum(s - v, 1.0e-30)
        out = tl.where(v >= 0.0, direct, conjugate)

    tl.store(O + offo, out, mask=mask)


@torch.no_grad()
def _launch(mode: int, x: torch.Tensor, y: torch.Tensor, y_obs: torch.Tensor | None,
            lam: float, alpha: float, beta: float, gamma: float) -> torch.Tensor:
    if not (math.isfinite(lam) and lam > 0.0):
        raise ValueError("lam must be finite and > 0")
    if not all(math.isfinite(z) for z in (alpha, beta, gamma)):
        raise ValueError("alpha, beta and gamma must be finite")
    out, placeholder, shape = _prepare(x, y, y_obs)
    sx = _strides4(x, shape)
    sy = _strides4(y, shape)
    syobs = _strides4(placeholder, shape)
    numel = out.numel()
    grid = (triton.cdiv(numel, 256),)
    _prox_axpy_kernel[grid](
        x, *sx, y, *sy, placeholder, *syobs, out,
        *shape,
        float(alpha), float(beta), float(gamma), float(lam),
        MODE=mode, BLOCK=256, num_warps=4,
    )
    return out


def prox_gaussian_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_GAUSSIAN, x, y, y_obs, lam, alpha, beta, gamma)


def prox_poisson_intensity_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_POISSON_INTENSITY, x, y, y_obs, lam, alpha, beta, gamma)


def prox_poisson_log_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_POISSON_LOG, x, y, y_obs, lam, alpha, beta, gamma)


def prox_poisson_nll_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    """Compatibility alias for the canonical log-intensity Poisson prox."""
    return prox_poisson_log_axpy(x, y, y_obs, lam, alpha, beta, gamma)


def prox_kl_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_KL, x, y, y_obs, lam, alpha, beta, gamma)


def prox_xlogx_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_XLOGX, x, y, None, lam, alpha, beta, gamma)


def prox_exp_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_EXP, x, y, None, lam, alpha, beta, gamma)


def prox_neglog_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _launch(MODE_NEGLOG, x, y, None, lam, alpha, beta, gamma)
