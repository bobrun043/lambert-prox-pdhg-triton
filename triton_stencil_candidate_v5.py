# triton_stencil_kernels_v5.py
#
# v5.16: Version "Stride-Safe"
# - CORRIGE (A): Ajout de 'VER' aux signatures de lancement (TypeError)
# - CORRIGE (B): Ajout de masques de halo stride-safe (Bug 99.6%)
# - (Conserve les correctifs v5.15 : Halley 3-branches, anti-annulation)

import torch
import triton
import triton.language as tl
import math

_HAS_TRITON = True

# --- Noyau 1: Fused Dual (grad2d + AXPY + prox_Gstar) ---

@triton.jit
def kernel_dual_tv_grad_fused(
    x_ptr,          # (N, 1, H, W)
    y_ptr,          # (N, 2, H, W)  [in]
    out_ptr,        # (N, 2, H, W)  [out]
    H: tl.constexpr, W: tl.constexpr,
    sigma, lam_tv,  # scalars
    # --- STRIDES (runtime scalars, pas constexpr !) ---
    STRIDE_X_N, STRIDE_X_C, STRIDE_X_H, STRIDE_X_W,
    STRIDE_Y_N, STRIDE_Y_C, STRIDE_Y_H, STRIDE_Y_W,
    STRIDE_OUT_N, STRIDE_OUT_C, STRIDE_OUT_H, STRIDE_OUT_W,
    VER: tl.constexpr,
    BLOCK_H: tl.constexpr = 32,
    BLOCK_W: tl.constexpr = 32,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # --- indices bloc
    ih = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    jw = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_cur = (ih[:, None] < H) & (jw[None, :] < W)

    # --- base ptrs
    x_base = x_ptr   + pid_n * STRIDE_X_N + 0 * STRIDE_X_C
    y_base = y_ptr   + pid_n * STRIDE_Y_N
    o_base = out_ptr + pid_n * STRIDE_OUT_N

    # --- offsets (H,W)
    off_x = ih[:, None] * STRIDE_X_H + jw[None, :] * STRIDE_X_W
    x_ij  = tl.load(x_base + off_x, mask=mask_cur, other=0.0)

    # halos (périodique) — RIGHT / DOWN
    jw_r = (pid_w * BLOCK_W + tl.arange(0, BLOCK_W) + 1) % W
    ih_d = (pid_h * BLOCK_H + tl.arange(0, BLOCK_H) + 1) % H

    mask_r = (ih[:, None] < H) & (jw_r[None, :] < W)
    off_r  = ih[:, None] * STRIDE_X_H + jw_r[None, :] * STRIDE_X_W
    x_r    = tl.load(x_base + off_r, mask=mask_r, other=0.0)

    mask_d = (ih_d[:, None] < H) & (jw[None, :] < W)
    off_d  = ih_d[:, None] * STRIDE_X_H + jw[None, :] * STRIDE_X_W
    x_d    = tl.load(x_base + off_d, mask=mask_d, other=0.0)

    gx = x_r - x_ij
    gy = x_d - x_ij

    # y old
    off_y = ih[:, None] * STRIDE_Y_H + jw[None, :] * STRIDE_Y_W
    y0_ij = tl.load(y_base + 0 * STRIDE_Y_C + off_y, mask=mask_cur, other=0.0)
    y1_ij = tl.load(y_base + 1 * STRIDE_Y_C + off_y, mask=mask_cur, other=0.0)

    y0_p = y0_ij + sigma * gx
    y1_p = y1_ij + sigma * gy

    n2 = y0_p * y0_p + y1_p * y1_p
    n  = tl.sqrt(tl.maximum(n2, 1e-30))
    s  = 1.0 / tl.maximum(1.0, n / lam_tv)
    y0_n = y0_p * s
    y1_n = y1_p * s

    # store avec STRIDES OUT
    off_o = ih[:, None] * STRIDE_OUT_H + jw[None, :] * STRIDE_OUT_W
    tl.store(o_base + 0 * STRIDE_OUT_C + off_o, y0_n, mask=mask_cur)
    tl.store(o_base + 1 * STRIDE_OUT_C + off_o, y1_n, mask=mask_cur)


@torch.no_grad()
def launch_dual_tv(x_bar, y_in, sigma, lam_tv, VER: int = 1516):
    if not _HAS_TRITON:
        raise ImportError("Triton non trouvé")

    assert x_bar.ndim == 4 and x_bar.shape[1] == 1
    assert y_in.ndim  == 4 and y_in.shape[1]  == 2
    N, C, H, W = x_bar.shape

    y_out = torch.empty_like(y_in)  # strides potentiellement ≠ contig

    BLOCK_H, BLOCK_W = (32, 32 if W <= 1024 else 64)
    grid = (N, triton.cdiv(H, BLOCK_H), triton.cdiv(W, BLOCK_W))

    kernel_dual_tv_grad_fused[grid](
        x_bar, y_in, y_out,
        H, W,
        float(sigma), float(lam_tv),
        # X strides
        x_bar.stride(0), x_bar.stride(1), x_bar.stride(2), x_bar.stride(3),
        # Y in strides
        y_in.stride(0), y_in.stride(1), y_in.stride(2), y_in.stride(3),
        # OUT strides (ne PAS réutiliser ceux de y_in)
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
        VER=VER,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8, num_stages=2,
    )
    return y_out


# --- Noyau 2: Fused Primal (div2d + AXPY + prox_F "Meta-Solveur") ---

# ============================================================
# PRIMAL (div backward, + prox_F) — STRIDE-SAFE
# ============================================================

@triton.jit
def _halley_u_log_u_fused(R, iters: tl.constexpr = 4):
    # solve u + log u = R  (stabilité v5.15)
    if R.dtype == tl.float16:
        eps = 1e-6
    elif R.dtype == tl.bfloat16:
        eps = 1e-6
    elif R.dtype == tl.float32:
        eps = 1e-12
    else:
        eps = 1e-18

    u_lo = tl.exp(R)
    u_hi = R - tl.log(R)
    u_md = 1.0 + R
    u_init = tl.where(R < -0.3, u_lo, tl.where(R > 8.0, u_hi, u_md))

    u = u_init
    for _ in range(iters):
        u = tl.maximum(u, eps)
        f   = u + tl.log(u) - R
        fp  = 1.0 + 1.0 / u
        fpp = -1.0 / (u * u)
        u  = u - (2.0 * f * fp) / (2.0 * fp * fp - f * fpp)
    return tl.maximum(u, eps)


@triton.jit
def kernel_primal_div_prox_fused(
    x_in_ptr,      # (N, 1, H, W)  in
    y_ptr,         # (N, 2, H, W)  in
    y_obs_ptr,     # (N, 1, H, W)  in (selon MODE)
    x_out_ptr,     # (N, 1, H, W)  out
    tau,           # scalar
    H: tl.constexpr, W: tl.constexpr,
    MODE: tl.constexpr,            # 0:L2 1:EXP 2:KL 3:NEGLOG 4:XLOGX 5:POISSON
    VER: tl.constexpr,
    # --- STRIDES IN (runtime) ---
    STRIDE_XIN_N, STRIDE_XIN_C, STRIDE_XIN_H, STRIDE_XIN_W,
    STRIDE_Y_N,   STRIDE_Y_C,    STRIDE_Y_H,   STRIDE_Y_W,
    STRIDE_YOBS_N, STRIDE_YOBS_C, STRIDE_YOBS_H, STRIDE_YOBS_W,
    # --- STRIDES OUT (runtime) ---
    STRIDE_XOUT_N, STRIDE_XOUT_C, STRIDE_XOUT_H, STRIDE_XOUT_W,
    BLOCK_H: tl.constexpr = 32,
    BLOCK_W: tl.constexpr = 32,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    ih = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    jw = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (ih[:, None] < H) & (jw[None, :] < W)

    # bases
    xin_base  = x_in_ptr   + pid_n * STRIDE_XIN_N + 0 * STRIDE_XIN_C
    xout_base = x_out_ptr  + pid_n * STRIDE_XOUT_N + 0 * STRIDE_XOUT_C
    y_base    = y_ptr      + pid_n * STRIDE_Y_N
    yobs_base = y_obs_ptr  + pid_n * STRIDE_YOBS_N + 0 * STRIDE_YOBS_C

    # X_in (i,j)
    off_xin = ih[:, None] * STRIDE_XIN_H + jw[None, :] * STRIDE_XIN_W
    xij     = tl.load(xin_base + off_xin, mask=mask, other=0.0)

    # Y courant
    off_y   = ih[:, None] * STRIDE_Y_H + jw[None, :] * STRIDE_Y_W
    yx_ij   = tl.load(y_base + 0 * STRIDE_Y_C + off_y, mask=mask, other=0.0)
    yy_ij   = tl.load(y_base + 1 * STRIDE_Y_C + off_y, mask=mask, other=0.0)

    # halos backward pour DIV: left/top (périodique)
    jw_l = (pid_w * BLOCK_W + tl.arange(0, BLOCK_W) - 1 + W) % W
    ih_t = (pid_h * BLOCK_H + tl.arange(0, BLOCK_H) - 1 + H) % H

    mask_l = (ih[:, None] < H) & (jw_l[None, :] < W)
    off_l  = ih[:, None] * STRIDE_Y_H + jw_l[None, :] * STRIDE_Y_W
    yx_l   = tl.load(y_base + 0 * STRIDE_Y_C + off_l, mask=mask_l, other=0.0)

    mask_t = (ih_t[:, None] < H) & (jw[None, :] < W)
    off_t  = ih_t[:, None] * STRIDE_Y_H + jw[None, :] * STRIDE_Y_W
    yy_t   = tl.load(y_base + 1 * STRIDE_Y_C + off_t, mask=mask_t, other=0.0)

    div = (yx_ij - yx_l) + (yy_ij - yy_t)
    z   = xij + tau * div

    # Prox routeur
    x_new = tl.zeros_like(z)

    if (MODE == 0) | (MODE == 2) | (MODE == 5):
        off_yobs = ih[:, None] * STRIDE_YOBS_H + jw[None, :] * STRIDE_YOBS_W
        yobs_ij  = tl.load(yobs_base + off_yobs, mask=mask, other=0.0)
    else:
        yobs_ij  = tl.zeros_like(z)

    if MODE == 0:        # L2 / Gaussian
        x_new = (z + tau * yobs_ij) / (1.0 + tau)

    elif MODE == 1:      # EXP
        R = z + tl.log(tau)
        u = _halley_u_log_u_fused(R)
        x_new = z - u

    elif MODE == 2:      # KL
        ysafe = tl.maximum(yobs_ij, 1e-30)
        R = (z + tau * tl.log(ysafe)) / tau - tl.log(tau)
        u = _halley_u_log_u_fused(R)
        x_new = tau * u

    elif MODE == 3:      # NEGLOG
        tiny = 1e-12 if z.dtype == tl.float32 else 1e-18
        disc = tl.sqrt(z * z + 4.0 * tau)
        tpos = 0.5 * (z + disc)
        talt = (2.0 * tau) / tl.maximum(disc - z, tiny)
        x_new = tl.where(z >= 0, tpos, talt)

    elif MODE == 4:      # XLOGX
        R = (z - tau) / tau - tl.log(tau)
        u = _halley_u_log_u_fused(R)
        x_new = tau * u

    elif MODE == 5:      # POISSON intensity
        tiny = 1e-12 if z.dtype == tl.float32 else 1e-18
        a    = z - tau
        ycl  = tl.maximum(yobs_ij, 0.0)
        disc = tl.sqrt(a * a + 4.0 * tau * ycl)
        tpos = 0.5 * (a + disc)
        talt = (2.0 * tau * ycl) / tl.maximum(disc - a, tiny)
        x_new = tl.where(a >= 0, tpos, talt)

    # store avec STRIDES OUT (NE PAS réutiliser ceux de X_in !)
    off_xout = ih[:, None] * STRIDE_XOUT_H + jw[None, :] * STRIDE_XOUT_W
    tl.store(xout_base + off_xout, x_new, mask=mask)


@torch.no_grad()
def launch_primal_div_prox(x_in, y, y_obs, tau, kind_str, x_out, VER: int = 1516):
    if not _HAS_TRITON:
        raise ImportError("Triton non trouvé")

    assert x_in.ndim == 4 and x_in.shape[1] == 1
    assert y.ndim    == 4 and y.shape[1]   == 2
    assert x_out.ndim == 4 and x_out.shape[1] == 1
    if x_in.data_ptr() == x_out.data_ptr():
        raise RuntimeError("x_in et x_out doivent être distincts (pas d'in-place).")

    N, C, H, W = x_in.shape

    MODE_MAP = {"gaussian":0, "l2":0, "exp":1, "kl":2, "neglog":3, "xlogx":4, "poisson_intensity":5}
    MODE = MODE_MAP.get(kind_str)
    if MODE is None:
        raise ValueError(f"Kind '{kind_str}' non supporté.")

    if y_obs is None:
        y_obs = x_in
    else:
        y_obs = y_obs.expand_as(x_in)

    BLOCK_H, BLOCK_W = (32, 32 if W <= 1024 else 64)
    grid = (N, triton.cdiv(H, BLOCK_H), triton.cdiv(W, BLOCK_W))

    kernel_primal_div_prox_fused[grid](
        # runtime scalaires
        x_in, y, y_obs, x_out, float(tau),
        # tl.constexpr selon la signature du kernel (AVANT strides)
        H, W, MODE, VER,
        # strides IN
        x_in.stride(0),  x_in.stride(1),  x_in.stride(2),  x_in.stride(3),
        y.stride(0),     y.stride(1),     y.stride(2),     y.stride(3),
        y_obs.stride(0), y_obs.stride(1), y_obs.stride(2), y_obs.stride(3),
        # strides OUT
        x_out.stride(0), x_out.stride(1), x_out.stride(2), x_out.stride(3),
        # constexpr restants
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8, num_stages=2,
    )
    return x_out


