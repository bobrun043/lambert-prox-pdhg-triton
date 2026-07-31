"""Compatibility surface routing legacy public names to the canonical backend.

This file contains no numerical implementation.  Its only purpose is to let
existing PDHG code migrate without keeping duplicate formulas.
"""

from __future__ import annotations

from typing import Any

import torch

from lambert_prox_torch_v1 import (
    prox_exp,
    prox_gaussian,
    prox_kl,
    prox_kl_log,
    prox_neglog,
    prox_poisson_intensity,
    prox_poisson_log,
    prox_xlogx,
    prox_xlogx_log,
    solve_u_log_u,
    solve_u_log_u_pair,
)


def solve_u_log_u_torch(R: torch.Tensor, iters: int = 6) -> torch.Tensor:
    return solve_u_log_u(R, iters=iters)


def prox_exp_torch(v: torch.Tensor, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_exp(v, lam, iters=iters)


def prox_xlogx_torch(v: torch.Tensor, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_xlogx(v, lam, iters=iters)


def prox_kl_torch(v: torch.Tensor, y_obs: Any, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_kl(v, y_obs, lam, iters=iters)


def prox_kl_log_torch(v: torch.Tensor, y_obs: Any, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_kl_log(v, y_obs, lam, iters=iters)


def prox_xlogx_log_torch(v: torch.Tensor, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_xlogx_log(v, lam, iters=iters)


def prox_poisson_torch(v: torch.Tensor, y_obs: Any, lam: Any) -> torch.Tensor:
    """Legacy name: Poisson in intensity coordinates."""
    return prox_poisson_intensity(v, y_obs, lam)


def prox_poisson_intensity_torch(v: torch.Tensor, y_obs: Any, lam: Any) -> torch.Tensor:
    return prox_poisson_intensity(v, y_obs, lam)


def prox_poisson_log_torch(v: torch.Tensor, y_obs: Any, lam: Any, iters: int = 6) -> torch.Tensor:
    return prox_poisson_log(v, y_obs, lam, iters=iters)


def prox_neglog_torch(v: torch.Tensor, lam: Any) -> torch.Tensor:
    return prox_neglog(v, lam)


def prox_gaussian_torch(v: torch.Tensor, y_obs: Any, lam: Any) -> torch.Tensor:
    return prox_gaussian(v, y_obs, lam)


__all__ = [
    "solve_u_log_u_pair",
    "solve_u_log_u_torch",
    "prox_exp_torch",
    "prox_xlogx_torch",
    "prox_kl_torch",
    "prox_kl_log_torch",
    "prox_xlogx_log_torch",
    "prox_poisson_torch",
    "prox_poisson_intensity_torch",
    "prox_poisson_log_torch",
    "prox_neglog_torch",
    "prox_gaussian_torch",
]
