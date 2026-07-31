"""Canonical PDHG integration for the Lambert-Prox project.

The module has one mathematical oracle for all separable fidelity proxes:
``lambert_prox_torch_v1``.  Accelerated backends are optional candidates and
are never selected silently.  Requesting an unavailable backend raises an
explicit error.

Problem class
-------------

    minimize_x F(x; y_obs) + lam_tv * TV_iso(Kx)

with a periodic forward finite-difference gradient K and K^T=-div.
V1 intentionally supports scalar images only: tensors (N,1,H,W).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Callable, Optional, Any
import importlib
import math
import platform
import time

import torch

from lambert_prox_torch_v1 import (
    prox_exp,
    prox_xlogx,
    prox_kl,
    prox_poisson_log,
    prox_poisson_intensity,
    prox_neglog,
    prox_gaussian,
)


class Backend(str, Enum):
    TORCH = "torch"
    TRITON_ELEMENTWISE = "triton_elementwise"
    TRITON_STENCIL = "triton_stencil"


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendReport:
    requested: str
    executed: str
    cuda_available: bool
    triton_importable: bool
    elementwise_module: str | None = None
    stencil_module: str | None = None


@dataclass(frozen=True)
class PDHGInfo:
    iterations: int
    converged: bool
    tau: float
    sigma: float
    norm_k: float
    backend: BackendReport
    primal_rel_change: float
    dual_rel_change: float
    elapsed_seconds: float
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _require_image(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"{name} must have shape (N,1,H,W); got {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64; got {x.dtype}")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def grad2d_periodic(x: torch.Tensor) -> torch.Tensor:
    """Forward periodic gradient Kx, output shape (N,2,H,W)."""
    _require_image(x, "x")
    dx = torch.roll(x, shifts=-1, dims=3) - x
    dy = torch.roll(x, shifts=-1, dims=2) - x
    return torch.cat((dx, dy), dim=1)


def div2d_periodic(p: torch.Tensor) -> torch.Tensor:
    """Backward periodic divergence; K^T p = -div(p)."""
    if not torch.is_tensor(p) or p.ndim != 4 or p.shape[1] != 2:
        raise ValueError("p must have shape (N,2,H,W)")
    px, py = p[:, :1], p[:, 1:]
    return (px - torch.roll(px, shifts=1, dims=3)) + (
        py - torch.roll(py, shifts=1, dims=2)
    )


def K(x: torch.Tensor) -> torch.Tensor:
    return grad2d_periodic(x)


def Kt(p: torch.Tensor) -> torch.Tensor:
    return -div2d_periodic(p)


def exact_norm_k_periodic_2d() -> float:
    return math.sqrt(8.0)


def check_adjoint(
    shape: tuple[int, int, int, int] = (2, 1, 31, 47),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    trials: int = 8,
    seed: int = 0,
) -> float:
    if shape[1] != 1:
        raise ValueError("canonical V1 requires C=1")
    g = torch.Generator(device=device).manual_seed(seed)
    worst = 0.0
    with torch.no_grad():
        for _ in range(trials):
            x = torch.randn(shape, generator=g, device=device, dtype=dtype)
            p = torch.randn((shape[0], 2, shape[2], shape[3]), generator=g, device=device, dtype=dtype)
            lhs = torch.sum(K(x) * p)
            rhs = torch.sum(x * Kt(p))
            scale = torch.maximum(torch.ones((), dtype=dtype, device=device), torch.maximum(lhs.abs(), rhs.abs()))
            worst = max(worst, float(((lhs - rhs).abs() / scale).item()))
    return worst


def prox_gstar_tv_iso(p: torch.Tensor, lam_tv: float) -> torch.Tensor:
    """Projection onto the pointwise l2 ball of radius lam_tv."""
    if lam_tv <= 0 or not math.isfinite(lam_tv):
        raise ValueError("lam_tv must be finite and > 0")
    px, py = p[:, :1], p[:, 1:]
    nrm = torch.sqrt(px * px + py * py)
    scale = torch.clamp(nrm / float(lam_tv), min=1.0)
    return torch.cat((px / scale, py / scale), dim=1)


def canonical_prox_f(
    kind: str,
    v: torch.Tensor,
    tau: float,
    y_obs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique Torch oracle for the primal fidelity prox."""
    k = kind.lower().strip()
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and > 0")
    if k in ("gaussian", "l2"):
        if y_obs is None:
            raise ValueError("gaussian requires y_obs")
        return prox_gaussian(v, y_obs, tau)
    if k == "poisson_intensity":
        if y_obs is None:
            raise ValueError("poisson_intensity requires y_obs")
        return prox_poisson_intensity(v, y_obs, tau)
    if k == "poisson_log":
        if y_obs is None:
            raise ValueError("poisson_log requires y_obs")
        return prox_poisson_log(v, y_obs, tau)
    if k == "kl":
        if y_obs is None:
            raise ValueError("kl requires y_obs")
        return prox_kl(v, y_obs, tau)
    if k == "xlogx":
        return prox_xlogx(v, tau)
    if k == "exp":
        return prox_exp(v, tau)
    if k == "neglog":
        return prox_neglog(v, tau)
    raise ValueError(f"unsupported fidelity kind: {kind!r}")



def validate_pdhg_model(kind: str, y_obs: torch.Tensor | None) -> None:
    """Reject fidelity/TV combinations that have no finite minimizer in V1.

    The prox of a function may be valid while the standalone variational model
    ``F + lam_tv*TV`` is not.  In particular, exp+TV has an unattained
    infimum along negative constants and -log+TV is unbounded below along
    positive constants.
    """
    k = kind.lower().strip()
    if k == "exp":
        raise ValueError("exp + TV alone has infimum 0 along x -> -infinity and no minimizer")
    if k == "neglog":
        raise ValueError("-log + TV alone is unbounded below along positive constants")
    if k == "poisson_log":
        if y_obs is None or not bool((y_obs > 0).all().item()):
            raise ValueError("poisson_log + TV V1 requires strictly positive y_obs for coercivity")
    if k in ("gaussian", "l2", "poisson_intensity", "kl") and y_obs is None:
        raise ValueError(f"{k} requires y_obs")
    if k == "kl" and not bool((y_obs > 0).all().item()):
        raise ValueError("KL model V1 requires strictly positive y_obs")
    if k == "poisson_intensity" and not bool((y_obs >= 0).all().item()):
        raise ValueError("poisson_intensity requires y_obs >= 0")

def _load_candidate_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise BackendUnavailableError(f"cannot import {name}: {exc}") from exc


def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(
    backend: Backend | str,
    *,
    elementwise_module: str = "triton_prox_fused_candidate_v2",
    stencil_module: str = "triton_stencil_candidate_v5",
) -> tuple[Backend, BackendReport, Any | None]:
    requested = Backend(backend)
    cuda = torch.cuda.is_available()
    tri = _triton_importable()
    if requested == Backend.TORCH:
        report = BackendReport(requested.value, Backend.TORCH.value, cuda, tri)
        return requested, report, None
    if not cuda:
        raise BackendUnavailableError(f"{requested.value} requested but CUDA is unavailable")
    if not tri:
        raise BackendUnavailableError(f"{requested.value} requested but Triton is not importable")
    if requested == Backend.TRITON_ELEMENTWISE:
        mod = _load_candidate_module(elementwise_module)
        report = BackendReport(
            requested.value,
            requested.value,
            cuda,
            tri,
            elementwise_module=elementwise_module,
        )
        return requested, report, mod
    mod = _load_candidate_module(stencil_module)
    report = BackendReport(
        requested.value,
        requested.value,
        cuda,
        tri,
        stencil_module=stencil_module,
    )
    return requested, report, mod


def _elementwise_candidate_prox(mod: Any, kind: str, x: torch.Tensor, kty: torch.Tensor,
                                y_obs: torch.Tensor | None, tau: float) -> torch.Tensor:
    kwargs = dict(lam=float(tau), alpha=1.0, beta=-float(tau), gamma=0.0)
    k = kind.lower().strip()
    if k in ("gaussian", "l2"):
        return mod.prox_gaussian_axpy(x, kty, y_obs, **kwargs)
    if k == "poisson_intensity":
        return mod.prox_poisson_intensity_axpy(x, kty, y_obs, **kwargs)
    if k == "poisson_log":
        return mod.prox_poisson_nll_axpy(x, kty, y_obs, **kwargs)
    if k == "kl":
        return mod.prox_kl_axpy(x, kty, y_obs, **kwargs)
    if k == "xlogx":
        return mod.prox_xlogx_axpy(x, kty, **kwargs)
    if k == "exp":
        return mod.prox_exp_axpy(x, kty, **kwargs)
    if k == "neglog":
        return mod.prox_neglog_axpy(x, kty, **kwargs)
    raise ValueError(kind)


def primal_update_torch(kind: str, x: torch.Tensor, y: torch.Tensor,
                        y_obs: torch.Tensor | None, tau: float) -> torch.Tensor:
    return canonical_prox_f(kind, x - float(tau) * Kt(y), float(tau), y_obs)


def dual_update_torch(x_bar: torch.Tensor, y: torch.Tensor,
                      sigma: float, lam_tv: float) -> torch.Tensor:
    return prox_gstar_tv_iso(y + float(sigma) * K(x_bar), lam_tv)


def pdhg_step(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    x_bar: torch.Tensor,
    y_obs: torch.Tensor | None,
    *,
    tau: float,
    sigma: float,
    lam_tv: float,
    theta: float = 1.0,
    backend: Backend | str = Backend.TORCH,
    candidate_module: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = Backend(backend)
    if selected == Backend.TORCH:
        y_new = dual_update_torch(x_bar, y, sigma, lam_tv)
        x_new = primal_update_torch(kind, x, y_new, y_obs, tau)
    elif selected == Backend.TRITON_ELEMENTWISE:
        if candidate_module is None:
            raise BackendUnavailableError("elementwise candidate module was not resolved")
        y_new = dual_update_torch(x_bar, y, sigma, lam_tv)
        kty = Kt(y_new)
        x_new = _elementwise_candidate_prox(candidate_module, kind, x, kty, y_obs, tau)
    else:
        if candidate_module is None:
            raise BackendUnavailableError("stencil candidate module was not resolved")
        y_new = candidate_module.launch_dual_tv(x_bar, y, float(sigma), float(lam_tv))
        out = torch.empty_like(x)
        x_new = candidate_module.launch_primal_div_prox(
            x, y_new, y_obs, float(tau), kind, out
        )
    x_bar_new = x_new + float(theta) * (x_new - x)
    return x_new, y_new, x_bar_new


def tv_iso(x: torch.Tensor) -> torch.Tensor:
    g = K(x)
    return torch.sqrt(g[:, :1] ** 2 + g[:, 1:] ** 2).sum()


def fidelity_value(kind: str, x: torch.Tensor, y_obs: torch.Tensor | None) -> torch.Tensor:
    k = kind.lower().strip()
    if k in ("gaussian", "l2"):
        if y_obs is None:
            raise ValueError("gaussian requires y_obs")
        return 0.5 * ((x - y_obs) ** 2).sum()
    if k == "poisson_intensity":
        if y_obs is None:
            raise ValueError("poisson_intensity requires y_obs")
        if bool((x <= 0).any().item()):
            return torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        return (x - y_obs * torch.log(x)).sum()
    if k == "poisson_log":
        if y_obs is None:
            raise ValueError("poisson_log requires y_obs")
        return (torch.exp(x) - y_obs * x).sum()
    if k == "kl":
        if y_obs is None:
            raise ValueError("kl requires y_obs")
        if bool((x < 0).any().item()) or bool((y_obs <= 0).any().item()):
            return torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        x_safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        return (x_safe * (torch.log(x_safe) - torch.log(y_obs)) - x_safe + y_obs).sum()
    if k == "xlogx":
        if bool((x < 0).any().item()):
            return torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        x_safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        return (x_safe * torch.log(x_safe)).sum()
    if k == "exp":
        return torch.exp(x).sum()
    if k == "neglog":
        if bool((x <= 0).any().item()):
            return torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        return (-torch.log(x)).sum()
    raise ValueError(kind)


def objective_value(kind: str, x: torch.Tensor, y_obs: torch.Tensor | None,
                    lam_tv: float) -> torch.Tensor:
    return fidelity_value(kind, x, y_obs) + float(lam_tv) * tv_iso(x)


def pdhg(
    kind: str,
    x0: torch.Tensor,
    y_obs: torch.Tensor | None,
    lam_tv: float,
    *,
    max_iter: int = 500,
    min_iter: int = 20,
    tol: float = 1e-5,
    theta: float = 1.0,
    tau: float | None = None,
    sigma: float | None = None,
    norm_k: float | None = None,
    backend: Backend | str = Backend.TORCH,
    monitor_every: int = 10,
    elementwise_module: str = "triton_prox_fused_candidate_v2",
    stencil_module: str = "triton_stencil_candidate_v5",
) -> tuple[torch.Tensor, torch.Tensor, PDHGInfo]:
    _require_image(x0, "x0")
    if y_obs is not None:
        _require_image(y_obs, "y_obs")
        if y_obs.shape != x0.shape or y_obs.device != x0.device or y_obs.dtype != x0.dtype:
            raise ValueError("y_obs must match x0 shape, device, and dtype")
    if lam_tv <= 0 or not math.isfinite(lam_tv):
        raise ValueError("lam_tv must be finite and > 0")
    validate_pdhg_model(kind, y_obs)
    if max_iter <= 0 or min_iter < 0 or min_iter > max_iter:
        raise ValueError("invalid iteration bounds")
    if tol < 0 or not math.isfinite(tol):
        raise ValueError("tol must be finite and >= 0")
    if monitor_every <= 0:
        raise ValueError("monitor_every must be positive")

    selected, backend_report, candidate = resolve_backend(
        backend, elementwise_module=elementwise_module, stencil_module=stencil_module
    )
    nk = exact_norm_k_periodic_2d() if norm_k is None else float(norm_k)
    if nk <= 0 or not math.isfinite(nk):
        raise ValueError("norm_k must be finite and > 0")
    tau_v = 0.99 / nk if tau is None else float(tau)
    sigma_v = 0.99 / nk if sigma is None else float(sigma)
    if tau_v <= 0 or sigma_v <= 0 or tau_v * sigma_v * nk * nk >= 1.0:
        raise ValueError("PDHG requires tau*sigma*||K||^2 < 1")

    x = x0.clone()
    y = torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
    x_bar = x.clone()
    history: dict[str, list[float]] = {"iteration": [], "primal_rel_change": [], "dual_rel_change": [], "objective": []}
    converged = False
    pr = float("inf")
    dr = float("inf")
    start = time.perf_counter()

    with torch.no_grad():
        for iteration in range(1, max_iter + 1):
            x_old, y_old = x, y
            x, y, x_bar = pdhg_step(
                kind, x, y, x_bar, y_obs,
                tau=tau_v, sigma=sigma_v, lam_tv=lam_tv, theta=theta,
                backend=selected, candidate_module=candidate,
            )
            pr = float(((x - x_old).norm() / x_old.norm().clamp_min(1.0)).item())
            dr = float(((y - y_old).norm() / y_old.norm().clamp_min(1.0)).item())
            if iteration == 1 or iteration % monitor_every == 0 or iteration == max_iter:
                obj = float(objective_value(kind, x, y_obs, lam_tv).item())
                history["iteration"].append(float(iteration))
                history["primal_rel_change"].append(pr)
                history["dual_rel_change"].append(dr)
                history["objective"].append(obj)
            if iteration >= min_iter and max(pr, dr) <= tol:
                converged = True
                break

    elapsed = time.perf_counter() - start
    info = PDHGInfo(
        iterations=iteration,
        converged=converged,
        tau=tau_v,
        sigma=sigma_v,
        norm_k=nk,
        backend=backend_report,
        primal_rel_change=pr,
        dual_rel_change=dr,
        elapsed_seconds=elapsed,
        history=history,
    )
    return x, y, info


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "triton_importable": _triton_importable(),
    }


__all__ = [
    "Backend", "BackendUnavailableError", "BackendReport", "PDHGInfo",
    "K", "Kt", "grad2d_periodic", "div2d_periodic", "check_adjoint",
    "exact_norm_k_periodic_2d", "prox_gstar_tv_iso", "canonical_prox_f", "validate_pdhg_model",
    "dual_update_torch", "primal_update_torch", "pdhg_step", "pdhg",
    "tv_iso", "fidelity_value", "objective_value", "environment_info",
]
