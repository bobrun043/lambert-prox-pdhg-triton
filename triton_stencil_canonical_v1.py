"""Canonical forward-only Triton fused PDHG stencils.

CUDA float32 candidate kernels matching ``pdhg_canonical_v1``.  They preserve
runtime strides, prohibit in-place primal updates, and support all seven
canonical fidelity proxes, including Poisson log-intensity.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

MODE_GAUSSIAN = 0
MODE_POISSON_INTENSITY = 1
MODE_POISSON_LOG = 2
MODE_KL = 3
MODE_XLOGX = 4
MODE_EXP = 5
MODE_NEGLOG = 6

MODE_MAP = {
    "gaussian": MODE_GAUSSIAN,
    "l2": MODE_GAUSSIAN,
    "poisson_intensity": MODE_POISSON_INTENSITY,
    "poisson_log": MODE_POISSON_LOG,
    "kl": MODE_KL,
    "xlogx": MODE_XLOGX,
    "exp": MODE_EXP,
    "neglog": MODE_NEGLOG,
}


def _check_image(x: torch.Tensor, name: str, channels: int) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError(f"{name} must be float32; got {x.dtype}")
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(f"{name} must have shape (N,{channels},H,W)")


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
    log_mask = R < -8.0
    Rq = tl.where(log_mask, R, -8.0)
    q = Rq
    for _ in range(ITERS):
        e = tl.exp(q)
        f = q + e - Rq
        fp = 1.0 + e
        den = 2.0 * fp * fp - f * e
        q = q - (2.0 * f * fp) / den

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
    return tl.where(log_mask, tl.exp(q), u), tl.where(log_mask, q, tl.log(u))


@triton.jit
def _dual_tv_kernel(
    X, Y, O,
    H: tl.constexpr, W: tl.constexpr,
    sigma: tl.float32, lam_tv: tl.float32,
    SXN: tl.int64, SXC: tl.int64, SXH: tl.int64, SXW: tl.int64,
    SYN: tl.int64, SYC: tl.int64, SYH: tl.int64, SYW: tl.int64,
    SON: tl.int64, SOC: tl.int64, SOH: tl.int64, SOW: tl.int64,
    BH: tl.constexpr, BW: tl.constexpr,
):
    n = tl.program_id(0)
    bh = tl.program_id(1)
    bw = tl.program_id(2)
    ih = bh * BH + tl.arange(0, BH)
    jw = bw * BW + tl.arange(0, BW)
    mask = (ih[:, None] < H) & (jw[None, :] < W)

    xb = X + n * SXN
    yb = Y + n * SYN
    ob = O + n * SON
    off = ih[:, None] * SXH + jw[None, :] * SXW
    x = tl.load(xb + off, mask=mask, other=0.0)

    jr = (jw + 1) % W
    idn = (ih + 1) % H
    xr = tl.load(xb + ih[:, None] * SXH + jr[None, :] * SXW, mask=mask, other=0.0)
    xd = tl.load(xb + idn[:, None] * SXH + jw[None, :] * SXW, mask=mask, other=0.0)

    offy = ih[:, None] * SYH + jw[None, :] * SYW
    yx = tl.load(yb + 0 * SYC + offy, mask=mask, other=0.0) + sigma * (xr - x)
    yy = tl.load(yb + 1 * SYC + offy, mask=mask, other=0.0) + sigma * (xd - x)
    nrm = tl.sqrt(tl.maximum(yx * yx + yy * yy, 1.0e-30))
    scale = 1.0 / tl.maximum(1.0, nrm / lam_tv)

    offo = ih[:, None] * SOH + jw[None, :] * SOW
    tl.store(ob + 0 * SOC + offo, yx * scale, mask=mask)
    tl.store(ob + 1 * SOC + offo, yy * scale, mask=mask)


@triton.jit
def _primal_kernel(
    X, Y, YOBS, O,
    tau: tl.float32,
    H: tl.constexpr, W: tl.constexpr,
    MODE: tl.constexpr,
    SXN: tl.int64, SXC: tl.int64, SXH: tl.int64, SXW: tl.int64,
    SYN: tl.int64, SYC: tl.int64, SYH: tl.int64, SYW: tl.int64,
    SZN: tl.int64, SZC: tl.int64, SZH: tl.int64, SZW: tl.int64,
    SON: tl.int64, SOC: tl.int64, SOH: tl.int64, SOW: tl.int64,
    BH: tl.constexpr, BW: tl.constexpr,
):
    n = tl.program_id(0)
    bh = tl.program_id(1)
    bw = tl.program_id(2)
    ih = bh * BH + tl.arange(0, BH)
    jw = bw * BW + tl.arange(0, BW)
    mask = (ih[:, None] < H) & (jw[None, :] < W)

    xb = X + n * SXN
    yb = Y + n * SYN
    zb = YOBS + n * SZN
    ob = O + n * SON

    offx = ih[:, None] * SXH + jw[None, :] * SXW
    x = tl.load(xb + offx, mask=mask, other=0.0)

    offy = ih[:, None] * SYH + jw[None, :] * SYW
    yx = tl.load(yb + 0 * SYC + offy, mask=mask, other=0.0)
    yy = tl.load(yb + 1 * SYC + offy, mask=mask, other=0.0)
    jl = (jw - 1 + W) % W
    it = (ih - 1 + H) % H
    yxl = tl.load(yb + 0 * SYC + ih[:, None] * SYH + jl[None, :] * SYW, mask=mask, other=0.0)
    yyt = tl.load(yb + 1 * SYC + it[:, None] * SYH + jw[None, :] * SYW, mask=mask, other=0.0)
    div = (yx - yxl) + (yy - yyt)
    v = x + tau * div  # x - tau*K^T y because K^T=-div
    log_tau = tl.log(tau)

    if MODE == 0:
        offz = ih[:, None] * SZH + jw[None, :] * SZW
        obs = tl.load(zb + offz, mask=mask, other=0.0)
        out = (v + tau * obs) / (1.0 + tau)

    elif MODE == 1:
        offz = ih[:, None] * SZH + jw[None, :] * SZW
        obs = tl.maximum(tl.load(zb + offz, mask=mask, other=0.0), 0.0)
        a = v - tau
        s = _hypot_pair(a, 2.0 * tl.sqrt(tau * obs))
        direct = 0.5 * (a + s)
        conjugate = 2.0 * tau * obs / tl.maximum(s - a, 1.0e-30)
        out = tl.where(a >= 0.0, direct, conjugate)

    elif MODE == 2:
        offz = ih[:, None] * SZH + jw[None, :] * SZW
        obs = tl.maximum(tl.load(zb + offz, mask=mask, other=0.0), 0.0)
        d = v + tau * obs
        _, q = _solve_u_log_u_pair(d + log_tau)
        out = q - log_tau

    elif MODE == 3:
        offz = ih[:, None] * SZH + jw[None, :] * SZW
        obs = tl.maximum(tl.load(zb + offz, mask=mask, other=0.0), 0.0)
        positive = obs > 0.0
        safe_obs = tl.where(positive, obs, 1.0)
        u, _ = _solve_u_log_u_pair(v / tau + tl.log(safe_obs) - log_tau)
        out = tl.where(positive, tau * u, 0.0)

    elif MODE == 4:
        u, _ = _solve_u_log_u_pair(v / tau - 1.0 - log_tau)
        out = tau * u

    elif MODE == 5:
        _, q = _solve_u_log_u_pair(v + log_tau)
        out = q - log_tau

    else:
        s = _hypot_pair(v, 2.0 * tl.sqrt(tau))
        direct = 0.5 * (v + s)
        conjugate = 2.0 * tau / tl.maximum(s - v, 1.0e-30)
        out = tl.where(v >= 0.0, direct, conjugate)

    offo = ih[:, None] * SOH + jw[None, :] * SOW
    tl.store(ob + offo, out, mask=mask)


@torch.no_grad()
def launch_dual_tv(x_bar: torch.Tensor, y_in: torch.Tensor, sigma: float, lam_tv: float):
    _check_image(x_bar, "x_bar", 1)
    _check_image(y_in, "y_in", 2)
    if x_bar.shape[0] != y_in.shape[0] or x_bar.shape[2:] != y_in.shape[2:]:
        raise ValueError("x_bar and y_in shapes are incompatible")
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be finite and > 0")
    if not (math.isfinite(lam_tv) and lam_tv > 0.0):
        raise ValueError("lam_tv must be finite and > 0")
    out = torch.empty_like(y_in, memory_format=torch.preserve_format)
    n, _, h, w = x_bar.shape
    bh, bw = 32, 32
    grid = (n, triton.cdiv(h, bh), triton.cdiv(w, bw))
    _dual_tv_kernel[grid](
        x_bar, y_in, out, h, w, float(sigma), float(lam_tv),
        *x_bar.stride(), *y_in.stride(), *out.stride(),
        BH=bh, BW=bw, num_warps=8,
    )
    return out


@torch.no_grad()
def launch_primal_div_prox(x_in: torch.Tensor, y: torch.Tensor,
                           y_obs: torch.Tensor | None, tau: float,
                           kind_str: str, x_out: torch.Tensor | None = None):
    _check_image(x_in, "x_in", 1)
    _check_image(y, "y", 2)
    if x_in.shape[0] != y.shape[0] or x_in.shape[2:] != y.shape[2:]:
        raise ValueError("x_in and y shapes are incompatible")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be finite and > 0")
    mode = MODE_MAP.get(kind_str.lower().strip())
    if mode is None:
        raise ValueError(f"unsupported fidelity kind: {kind_str!r}")
    needs_obs = mode in {MODE_GAUSSIAN, MODE_POISSON_INTENSITY, MODE_POISSON_LOG, MODE_KL}
    if needs_obs:
        if y_obs is None:
            raise ValueError(f"{kind_str} requires y_obs")
        _check_image(y_obs, "y_obs", 1)
        if y_obs.shape != x_in.shape:
            raise ValueError("y_obs must have the same shape as x_in")
        placeholder = y_obs
    else:
        placeholder = x_in

    if x_out is None:
        x_out = torch.empty_like(x_in, memory_format=torch.preserve_format)
    else:
        _check_image(x_out, "x_out", 1)
        if x_out.shape != x_in.shape or x_out.device != x_in.device:
            raise ValueError("x_out must match x_in")
    if x_out.data_ptr() == x_in.data_ptr():
        raise RuntimeError("in-place primal updates are forbidden")

    n, _, h, w = x_in.shape
    bh, bw = 32, 32
    grid = (n, triton.cdiv(h, bh), triton.cdiv(w, bw))
    _primal_kernel[grid](
        x_in, y, placeholder, x_out, float(tau), h, w, mode,
        *x_in.stride(), *y.stride(), *placeholder.stride(), *x_out.stride(),
        BH=bh, BW=bw, num_warps=8,
    )
    return x_out
