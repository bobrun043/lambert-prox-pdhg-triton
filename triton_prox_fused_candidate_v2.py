# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:25:37 2025

@author: laurent
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 00:32:48 2025

@author: laurent
"""

# triton_prox_fused_v2.py
# MODIFIÉ: Autotune + 2 noyaux manquants (neglog, gaussian)
import math
import torch
import triton
import triton.language as tl

# ---------- util ND: shape/strides en 4D (broadcast: stride=0 si dim=1) ----------
def _as4(shape):
    s = (1, 1, 1, 1)
    return (s[:4-len(shape)] + tuple(shape)) if len(shape) < 4 else tuple(shape[-4:])

def _strides4(t, out_shape4):
    if t is None:
        return (0, 0, 0, 0)
    st = t.stride()
    sh = t.shape
    st4 = (0, 0, 0, 0)
    st4 = (st4[:4-len(st)] + tuple(st)) if len(st) < 4 else tuple(st[-4:])
    sh4 = _as4(sh)
    return tuple(st4[i] if sh4[i] == out_shape4[i] else 0 for i in range(4))

def _broadcast_shape(*shapes):
    out = ()
    for dims in zip(*[ ( (1,)*(4-len(s)) + tuple(s) ) if len(s)<4 else tuple(s[-4:]) for s in shapes if s is not None ]):
        m = max(dims)
        if any((d not in (1, m)) for d in dims):
            raise ValueError("Shapes non broadcastables: " + str(shapes))
        out += (m,)
    return out

def _prep_out(x, y=None, yobs=None, dtype=None, like=None):
    sh = _broadcast_shape(*(t.shape for t in (x, y, yobs) if t is not None))
    if like is not None:
        out = torch.empty(sh, dtype=like.dtype, device=like.device, layout=torch.strided)
    else:
        dd = dtype if dtype is not None else (x.dtype if x is not None else torch.float32)
        dev = x.device if x is not None else (y.device if y is not None else yobs.device)
        out = torch.empty(sh, dtype=dd, device=dev, layout=torch.strided)
    return out

# --- MODIFIÉ: Ajout de la config Autotune ---
AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK': 2048}, num_warps=4),
    triton.Config({'BLOCK': 4096}, num_warps=4),
    triton.Config({'BLOCK': 4096}, num_warps=8),
    triton.Config({'BLOCK': 8192}, num_warps=8),
]

# ---------- Halley pour résoudre u + log(u) = R (log-domaine, overflow-safe) ----------
@triton.jit
def _solve_u_log_u_halley(R, eps):
    # --- CORRECTION: Amorce alignée sur solve_u_log_u_torch (3 branches) ---
    # R < -0.3
    u_lo = tl.exp(R)
    # R > 8.0
    u_hi = R - tl.log(R) # R > 8.0, donc log(R) est sûr
    # -0.3 <= R <= 8.0
    u_md = 1.0 + R
    
    # Implémentation via tl.where imbriqués
    u_md_hi = tl.where(R > 8.0, u_hi, u_md)
    w = tl.where(R < -0.3, u_lo, u_md_hi)
    # --- FIN DE LA CORRECTION ---

    for _ in range(4):
        w = tl.maximum(w, eps)
        f  = w + tl.log(w) - R
        fp = 1.0 + 1.0 / w
        fpp= -1.0 / (w * w)
        num = 2.0 * f * fp
        den = 2.0 * fp * fp - f * fpp
        step = num / den
        w = w - step
    return tl.maximum(w, eps)

# --- MODIFIÉ: Suppression de _grid (géré par autotune) ---

# ---------- prox_exp: t = v - W0(λ e^v), backward: dt/dv = 1/(1+λ e^t) ----------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_exp_axpy(X, SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                        Y, SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                        O, SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                        N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                        alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                        lam:tl.float32, eps:tl.float32,
                        numel:tl.int32, # --- MODIFIÉ ---
                        BLOCK:tl.constexpr): # --- MODIFIÉ (gardé pour autotune) ---
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0 = N1*N2*N3; S1 = N2*N3; S2 = N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)

    v = alpha * x + beta * y + gamma
    L = tl.log(lam) + v
    w = _solve_u_log_u_halley(L, eps)
    t = v - w
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)

# ---------- prox_xlogx: solve t + λ log t = v - λ ----------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_xlogx_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                          Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                          O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                          N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                          alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                          lam:tl.float32, eps:tl.float32,
                          numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0 = N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    v = alpha * x + beta * y + gamma

    R = (v - lam) / lam - tl.log(lam)
    u = _solve_u_log_u_halley(R, eps)
    t = tl.maximum(lam * u, eps)
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)

# ---------- prox_KL: solve t + λ log t = v + λ log y_obs ----------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_kl_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                       Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                       YOBS, SYO0:tl.int32,SYO1:tl.int32,SYO2:tl.int32,SYO3:tl.int32,
                       O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                       N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                       alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                       lam:tl.float32, eps:tl.float32,
                       numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0=N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3
    offoy= i0*SYO0+ i1*SYO1+ i2*SYO2+ i3*SYO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    yobs = tl.cast(tl.load(YOBS + offoy, mask=mask, other=1.), tl.float32)
    yobs = tl.maximum(yobs, eps)

    v = alpha * x + beta * y + gamma
    R = (v + lam * tl.log(yobs)) / lam - tl.log(lam)
    u = _solve_u_log_u_halley(R, eps)
    t = tl.maximum(lam * u, eps)
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)

# ---------- prox_PoissonNLL: (v-λ)+sqrt((v-λ)^2 + 4λy) -- ERREUR DANS LE COMMENTAIRE ORIGINAL --
# Le prox de Poisson NLL est: 1/2 (x-v)^2 + lam (e^x - y x)
# L'équation est: x - v + lam (e^x - y) = 0
# t = d - W0(lam * exp(d)) avec d = v + lam*y
# Le commentaire original décrivait prox_NEG_LOG, pas Poisson.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_poisson_NLL_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                            Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                            YOBS,SYO0:tl.int32,SYO1:tl.int32,SYO2:tl.int32,SYO3:tl.int32,
                            O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                            N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                            alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                            lam:tl.float32, eps:tl.float32,
                            numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0=N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3
    offoy= i0*SYO0+ i1*SYO1+ i2*SYO2+ i3*SYO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    yobs = tl.cast(tl.load(YOBS + offoy, mask=mask, other=0.), tl.float32)
    yobs = tl.maximum(yobs, 0.)

    v = alpha * x + beta * y + gamma
    
    # --- NON, LE PROX POISSON VIENT DE solve_u_log_u ---
    # C'était prox_neglog qui avait la forme fermée.
    # Le prox de Poisson NLL (e^x - yx) est:
    # d = v + lam*yobs
    # L = log(lam) + d
    # w = solve_u_log_u(L)
    # t = d - w
    d = v + lam * yobs
    L = tl.log(lam) + d
    w = _solve_u_log_u_halley(L, eps)
    t = d - w
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)

# --- AJOUT: prox_neglog: t = 0.5*((v)+sqrt(v^2 + 4λ)) ---
# L'équation est: x - v - lam/x = 0
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_neglog_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                           Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                           O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                           N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                           alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                           lam:tl.float32, eps:tl.float32,
                           numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0=N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    v = alpha * x + beta * y + gamma
    
    disc = tl.sqrt(v*v + 4.0*lam)
    t = 0.5 * (v + disc)
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)


# --- AJOUT: prox_gaussian: t = (v + λ*y_obs) / (1 + λ) ---
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_gaussian_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                             Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                             YOBS, SYO0:tl.int32,SYO1:tl.int32,SYO2:tl.int32,SYO3:tl.int32,
                             O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                             N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                             alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                             lam:tl.float32, eps:tl.float32,
                             numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    S0=N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3
    offoy= i0*SYO0+ i1*SYO1+ i2*SYO2+ i3*SYO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    yobs = tl.cast(tl.load(YOBS + offoy, mask=mask, other=0.), tl.float32)

    v = alpha * x + beta * y + gamma
    t = (v + lam * yobs) / (1.0 + lam)
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)


# --- AJOUT: prox_gaussian: t = (v + λ*y_obs) / (1 + λ) ---
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['numel'])
@triton.jit
def _kern_prox_poisson_INTENSITY_axpy(X,SX0:tl.int32,SX1:tl.int32,SX2:tl.int32,SX3:tl.int32,
                            Y,SY0:tl.int32,SY1:tl.int32,SY2:tl.int32,SY3:tl.int32,
                            YOBS,SYO0:tl.int32,SYO1:tl.int32,SYO2:tl.int32,SYO3:tl.int32,
                            O,SO0:tl.int32,SO1:tl.int32,SO2:tl.int32,SO3:tl.int32,
                            N0:tl.int32,N1:tl.int32,N2:tl.int32,N3:tl.int32,
                            alpha:tl.float32,beta:tl.float32,gamma:tl.float32,
                            lam:tl.float32, eps:tl.float32,
                            numel:tl.int32, BLOCK:tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    # ... (calcul des offsets inchangé) ...
    # i0, i1, i2, i3, offx, offy, offo, offoy
    S0=N1*N2*N3; S1=N2*N3; S2=N3
    i0 = off // S0; r  = off - i0*S0
    i1 = r // S1;   r  = r - i1*S1
    i2 = r // S2;   i3 = r - i2*S2

    offx = i0*SX0 + i1*SX1 + i2*SX2 + i3*SX3
    offy = i0*SY0 + i1*SY1 + i2*SY2 + i3*SY3
    offo = i0*SO0 + i1*SO1 + i2*SO2 + i3*SO3
    offoy= i0*SYO0+ i1*SYO1+ i2*SYO2+ i3*SYO3

    x = tl.cast(tl.load(X + offx, mask=mask, other=0.), tl.float32)
    y = tl.cast(tl.load(Y + offy, mask=mask, other=0.), tl.float32)
    yobs = tl.cast(tl.load(YOBS + offoy, mask=mask, other=0.), tl.float32)
    yobs = tl.maximum(yobs, 0.)

    v = alpha * x + beta * y + gamma
    
    # --- MODIFICATION (Forme fermée pour x - y*log(x)) ---
    a = v - lam
    disc = tl.sqrt(a*a + 4.0*lam*yobs)
    
    # t = 0.5 * (a + disc)
    t1 = 0.5 * (a + disc);
    t2 = (2.0 * lam * yobs) / (disc - a + eps);
    t  = tl.where(a >= 0, t1, t2);
    # --- ANCIENNE FORME (Lambert pour e^x - y*x) ---
    # d = v + lam * yobs
    # L = tl.log(lam) + d
    # w = _solve_u_log_u_halley(L, eps)
    # t = d - w
    # --- FIN MODIFICATION ---
    
    tl.store(O + offo, tl.cast(t, tl.float32), mask=mask)

# ---------- wrappers autograd ----------
def _eps(dtype):
    return 1e-6 if dtype in (torch.float16, torch.bfloat16) else 1e-12

class _ProxExpAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, lam, alpha, beta, gamma):
        assert x.device.type == 'cuda'
        out = _prep_out(x, y, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        SX = _strides4(x, (N0,N1,N2,N3))
        SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3))
        
        # --- MODIFIÉ: Lancement Autotune ---
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_exp_axpy[grid](
            x, *SX, y, *SY, out, *SO, N0,N1,N2,N3,
            float(alpha), float(beta), float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)
        # --- FIN MODIFICATION ---
            
        ctx.save_for_backward(out, x, y)
        ctx.params = (lam, alpha, beta, gamma)
        return out
    @staticmethod
    def backward(ctx, gO):
        t, x, y = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        w = lam * torch.exp(t)
        dv = gO * (1.0 / (1.0 + w))
        gx = dv * alpha
        gy = dv * beta
        glam = (gO * (- w / (lam * (1.0 + w)))).sum()
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()
        return gx, gy, glam, galpha, gbeta, ggamma

class _ProxXLogXAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, lam, alpha, beta, gamma):
        out = _prep_out(x, y, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        SX = _strides4(x, (N0,N1,N2,N3))
        SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3))
        
        # --- MODIFIÉ: Lancement Autotune ---
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_xlogx_axpy[grid](
            x,*SX, y,*SY, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)
        # --- FIN MODIFICATION ---

        ctx.save_for_backward(out, x, y)
        ctx.params = (lam, alpha, beta, gamma)
        return out
    @staticmethod
    def backward(ctx, gO):
        t, x, y = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        dv = gO * (t / (t + lam))
        gx = dv * alpha
        gy = dv * beta
        glam = (gO * ( - (1.0 + torch.log(torch.clamp_min(t, 1e-30))) * t / (t + lam) )).sum()
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()
        return gx, gy, glam, galpha, gbeta, ggamma

class _ProxKLAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, yobs, lam, alpha, beta, gamma):
        out = _prep_out(x, y, yobs, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        SX = _strides4(x, (N0,N1,N2,N3))
        SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3))
        SYO= _strides4(yobs, (N0,N1,N2,N3))
        
        # --- MODIFIÉ: Lancement Autotune ---
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_kl_axpy[grid](
            x,*SX, y,*SY, yobs,*SYO, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)
        # --- FIN MODIFICATION ---
            
        ctx.save_for_backward(out, x, y, yobs)
        ctx.params = (lam, alpha, beta, gamma)
        return out
    @staticmethod
    def backward(ctx, gO):
        t, x, y, yobs = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        t = torch.clamp_min(t, 1e-30); yobs = torch.clamp_min(yobs, 1e-30)
        dv = gO * (t / (t + lam))
        gx = dv * alpha
        gy = dv * beta
        gyobs = gO * (lam * t / (yobs * (t + lam)))
        glam = (gO * ( - t * torch.log(t / yobs) / (t + lam) )).sum()
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()
        return gx, gy, gyobs, glam, galpha, gbeta, ggamma


class _ProxPoissonNLLAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, yobs, lam, alpha, beta, gamma):
        out = _prep_out(x, y, yobs, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        # ... (calcul des strides SX, SY, SYO, SO) ...
        SX = _strides4(x, (N0,N1,N2,N3)); SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3)); SYO= _strides4(yobs, (N0,N1,N2,N3))

        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_poisson_NLL_axpy[grid](
            x,*SX, y,*SY, yobs,*SYO, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)

        ctx.save_for_backward(x, y, yobs, out) # Sauve t (out)
        ctx.params = (lam, alpha, beta, gamma)
        return out

    @staticmethod
    def backward(ctx, gO):
        x, y, yobs, t = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params

        # Reconstruit v et w (interne au prox)
        # t = (v + lam*yobs) - w  =>  w = v + lam*yobs - t
        v = (alpha*x + beta*y + gamma)
        w = v + lam*yobs - t

        # dt/dz = 1 / (1 + w)
        denom = 1.0 / (1.0 + w)

        # dt/dv = dt/dz * dz/dv = denom * 1
        dv = gO * denom

        gx = dv * alpha
        gy = dv * beta

        # dt/dyobs = dt/dz * dz/dyobs = denom * lam
        gyobs = gO * (lam * denom)

        # dt/dlam = dt/dz * dz/dlam + dt/dw * dw/dlam ... (plus complexe)
        # Simplifié : on ne dérive pas /lam pour le PDHG
        glam = torch.tensor(0.0, device=gO.device) 
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()

        return gx, gy, gyobs, glam, galpha, gbeta, ggamma


class _ProxPoissonINTENSITYAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, yobs, lam, alpha, beta, gamma):
        out = _prep_out(x, y, yobs, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        # ... (calcul des strides SX, SY, SYO, SO) ...
        SX = _strides4(x, (N0,N1,N2,N3)); SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3)); SYO= _strides4(yobs, (N0,N1,N2,N3))

        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_poisson_INTENSITY_axpy[grid]( # <-- Appel au bon noyau
            x,*SX, y,*SY, yobs,*SYO, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)

        ctx.save_for_backward(out, x, y, yobs)
        ctx.params = (lam, alpha, beta, gamma)
        return out

    @staticmethod
    def backward(ctx, gO):
        # Le backward original (lignes 433-467) est déjà correct pour
        # la forme fermée "intensité"
        t, x, y, yobs = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        a = (alpha * x + beta * y + gamma) - lam
        disc = torch.sqrt(a*a + 4.0*lam*torch.clamp_min(yobs, 0.0))

        dt_dv = 0.5 * (1.0 + a / disc)
        dt_dyobs = lam / disc
        dv = gO * dt_dv

        gx = dv * alpha
        gy = dv * beta
        gyobs = gO * dt_dyobs

        dt_dlam = 0.5 * (-1.0 + (-a + 2.0*yobs)/disc)
        glam = (gO * dt_dlam).sum()

        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()

        return gx, gy, gyobs, glam, galpha, gbeta, ggamma



# --- AJOUT: Classe Autograd pour prox_neglog ---
class _ProxNeglogAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, lam, alpha, beta, gamma):
        out = _prep_out(x, y, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        SX = _strides4(x, (N0,N1,N2,N3))
        SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3))
        
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_neglog_axpy[grid](
            x,*SX, y,*SY, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)
            
        ctx.save_for_backward(out, x, y)
        ctx.params = (lam, alpha, beta, gamma)
        return out
    @staticmethod
    def backward(ctx, gO):
        t, x, y = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        # t = 0.5 * (v + sqrt(v^2 + 4*lam))
        # dt/dv = 0.5 * (1 + v / sqrt(v^2 + 4*lam))
        v = (alpha*x + beta*y + gamma)
        disc = torch.sqrt(v*v + 4.0*lam)
        dt_dv = 0.5 * (1.0 + v / disc)
        
        dv = gO * dt_dv
        gx = dv * alpha
        gy = dv * beta
        # dt/dlam = 0.5 * (2 / disc) = 1 / disc
        glam = (gO * (1.0 / disc)).sum()
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()
        return gx, gy, glam, galpha, gbeta, ggamma

# --- AJOUT: Classe Autograd pour prox_gaussian ---
class _ProxGaussianAXPY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, yobs, lam, alpha, beta, gamma):
        out = _prep_out(x, y, yobs, dtype=torch.float32, like=x)
        N0,N1,N2,N3 = _as4(out.shape)
        SX = _strides4(x, (N0,N1,N2,N3))
        SY = _strides4(y, (N0,N1,N2,N3))
        SO = _strides4(out, (N0,N1,N2,N3))
        SYO= _strides4(yobs, (N0,N1,N2,N3))
        
        numel = out.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
        _kern_prox_gaussian_axpy[grid](
            x,*SX, y,*SY, yobs,*SYO, out,*SO, N0,N1,N2,N3,
            float(alpha),float(beta),float(gamma), float(lam), float(_eps(out.dtype)),
            numel=numel)
            
        ctx.save_for_backward(out, x, y, yobs)
        ctx.params = (lam, alpha, beta, gamma)
        return out
    @staticmethod
    def backward(ctx, gO):
        t, x, y, yobs = ctx.saved_tensors
        lam, alpha, beta, gamma = ctx.params
        # t = (v + lam*yobs) / (1.0 + lam)
        inv_1_plus_lam = 1.0 / (1.0 + lam)
        
        dt_dv = inv_1_plus_lam
        dv = gO * dt_dv
        
        gx = dv * alpha
        gy = dv * beta
        gyobs = gO * (lam * inv_1_plus_lam)
        # dt/dlam = (yobs*(1+lam) - (v+lam*yobs)) / (1+lam)^2
        #         = (yobs - t) / (1+lam)
        dt_dlam = (yobs - t) * inv_1_plus_lam
        glam = (gO * dt_dlam).sum()
        galpha = (dv * x).sum()
        gbeta  = (dv * y).sum()
        ggamma = dv.sum()
        return gx, gy, gyobs, glam, galpha, gbeta, ggamma

# ---------- API ----------
def prox_exp_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _ProxExpAXPY.apply(x, y, float(lam), float(alpha), float(beta), float(gamma))

def prox_xlogx_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _ProxXLogXAXPY.apply(x, y, float(lam), float(alpha), float(beta), float(gamma))

def prox_kl_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _ProxKLAXPY.apply(x, y, y_obs, float(lam), float(alpha), float(beta), float(gamma))

def prox_poisson_nll_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    """
    Prox de f(t) = e^t - y*t (NLL Canonique).
    Utilise le solveur de Halley (Classe Lambert).
    """
    return _ProxPoissonNLLAXPY.apply(x, y, y_obs, float(lam), float(alpha), float(beta), float(gamma))

def prox_poisson_intensity_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    """
    Prox de f(x) = x - y*log(x) (NLL Intensité).
    Utilise la forme fermée (quadratique).
    """
    return _ProxPoissonINTENSITYAXPY.apply(x, y, y_obs, float(lam), float(alpha), float(beta), float(gamma))

# --- AJOUT: API pour neglog et gaussian ---
def prox_neglog_axpy(x, y, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _ProxNeglogAXPY.apply(x, y, float(lam), float(alpha), float(beta), float(gamma))

def prox_gaussian_axpy(x, y, y_obs, lam, alpha=1.0, beta=1.0, gamma=0.0):
    return _ProxGaussianAXPY.apply(x, y, y_obs, float(lam), float(alpha), float(beta), float(gamma))

# ---------- mini-bench (Corrigé pour NLL vs Intensité) ----------
@torch.inference_mode()
def bench(kind="exp", n=2_000_000, repeats=10, dtype=torch.float32, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(0)
    def rand(shape): return torch.randn(shape, dtype=dtype, device=device, generator=gen)
    x = rand((n,)); y = rand((n,))
    yobs_pos = torch.rand((n,), dtype=dtype, device=device, generator=gen) + 0.1
    lam = 0.7; a=1.1; b=0.9; g=-0.2
    torch.cuda.synchronize()
    
    # --- Début de la Correction ---
    
    # warmup
    if kind=="exp": 
        prox_exp_axpy(x,y,lam,a,b,g); torch.cuda.synchronize()
    if kind=="xlogx": 
        prox_xlogx_axpy(x,y,lam,a,b,g); torch.cuda.synchronize()
    if kind=="kl": 
        prox_kl_axpy(x,y,yobs_pos,lam,a,b,g); torch.cuda.synchronize()
    if kind=="poisson_nll": 
        prox_poisson_nll_axpy(x,y,yobs_pos,lam,a,b,g); torch.cuda.synchronize()
    if kind=="poisson_intensity": 
        prox_poisson_intensity_axpy(x,y,yobs_pos,lam,a,b,g); torch.cuda.synchronize()
    if kind=="neglog": 
        prox_neglog_axpy(x,y,lam,a,b,g); torch.cuda.synchronize()
    if kind=="gaussian": 
        prox_gaussian_axpy(x,y,yobs_pos,lam,a,b,g); torch.cuda.synchronize()
        
    # time
    import time
    tmin = 1e9
    for _ in range(repeats):
        torch.cuda.synchronize(); t0=time.perf_counter()
        
        if kind=="exp": prox_exp_axpy(x,y,lam,a,b,g)
        elif kind=="xlogx": prox_xlogx_axpy(x,y,lam,a,b,g)
        elif kind=="kl": prox_kl_axpy(x,y,yobs_pos,lam,a,b,g)
        elif kind=="poisson_nll": prox_poisson_nll_axpy(x,y,yobs_pos,lam,a,b,g)
        elif kind=="poisson_intensity": prox_poisson_intensity_axpy(x,y,yobs_pos,lam,a,b,g)
        elif kind=="neglog": prox_neglog_axpy(x,y,lam,a,b,g)
        elif kind=="gaussian": prox_gaussian_axpy(x,y,yobs_pos,lam,a,b,g)
            
        torch.cuda.synchronize(); dt=time.perf_counter()-t0
        tmin = min(tmin, dt)
        
    # --- Fin de la Correction ---
        
    ns_per_elt = (tmin * 1e9) / n
    mops = (n / tmin) / 1e6
    return ns_per_elt, mops

# ---------- Lancement du mini-bench ----------
if __name__ == "__main__":
    print("--- Démarrage du mini-bench (Triton Elementwise v2 corrigé) ---")
    
    # Liste des 7 prox désormais disponibles
    kinds_to_test = [
        "gaussian", 
        "exp", 
        "xlogx", 
        "kl", 
        "neglog", 
        "poisson_nll", 
        "poisson_intensity"
    ]
    
    print(f"{'Kind':<20} | {'ns/elt':<10} | {'Mops/s':<10}")
    print("-" * 44)
    
    for kind in kinds_to_test:
        try:
            # Appel de la fonction bench pour chaque 'kind'
            ns, mops = bench(kind=kind, n=2_000_000, repeats=10, device="cuda")
            print(f"{kind:<20} | {ns:<10.1f} | {mops:<10.1f}")
        except Exception as e:
            print(f"{kind:<20} | ERREUR: {e}")
            
    print("--- Mini-bench terminé ---")