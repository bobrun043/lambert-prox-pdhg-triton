import math
import pytest
import torch

from lambert_prox_torch_v1 import (
    prox_exp, prox_xlogx, prox_kl, prox_poisson_log,
    prox_poisson_intensity, prox_neglog, prox_gaussian,
)
from pdhg_canonical_v1 import (
    Backend, BackendUnavailableError, K, Kt, check_adjoint,
    exact_norm_k_periodic_2d, prox_gstar_tv_iso, canonical_prox_f,
    dual_update_torch, primal_update_torch, pdhg_step, pdhg,
    objective_value, validate_pdhg_model,
)


def tensors(dtype=torch.float64):
    g = torch.Generator().manual_seed(42)
    x = torch.randn((2,1,17,23), generator=g, dtype=dtype)
    y = torch.randn((2,2,17,23), generator=g, dtype=dtype)
    obs = torch.rand((2,1,17,23), generator=g, dtype=dtype) + 0.2
    return x, y, obs


def test_adjoint_periodic_fp64():
    assert check_adjoint() < 5e-14


def test_exact_norm_contract():
    assert exact_norm_k_periodic_2d() == pytest.approx(math.sqrt(8.0), rel=0, abs=0)


def test_tv_dual_projection_contract():
    _, p, _ = tensors()
    lam = 0.17
    out = prox_gstar_tv_iso(p, lam)
    nrm = torch.sqrt(out[:, :1] ** 2 + out[:, 1:] ** 2)
    assert float(nrm.max()) <= lam * (1 + 2e-15)


@pytest.mark.parametrize("kind", [
    "gaussian", "poisson_intensity", "poisson_log", "kl",
    "xlogx", "exp", "neglog",
])
def test_canonical_factory_equals_direct(kind):
    x, _, obs = tensors()
    tau = 0.37
    yobs = obs if kind in {"gaussian","poisson_intensity","poisson_log","kl"} else None
    got = canonical_prox_f(kind, x, tau, yobs)
    direct = {
        "gaussian": lambda: prox_gaussian(x, obs, tau),
        "poisson_intensity": lambda: prox_poisson_intensity(x, obs, tau),
        "poisson_log": lambda: prox_poisson_log(x, obs, tau),
        "kl": lambda: prox_kl(x, obs, tau),
        "xlogx": lambda: prox_xlogx(x, tau),
        "exp": lambda: prox_exp(x, tau),
        "neglog": lambda: prox_neglog(x, tau),
    }[kind]()
    torch.testing.assert_close(got, direct, rtol=0, atol=0)


@pytest.mark.parametrize("kind", [
    "gaussian", "poisson_intensity", "poisson_log", "kl",
    "xlogx", "exp", "neglog",
])
def test_one_step_equals_explicit_oracle(kind):
    x, y, obs = tensors()
    xbar = x + 0.1
    tau, sigma, lam_tv = 0.2, 0.3, 0.07
    yobs = obs if kind in {"gaussian","poisson_intensity","poisson_log","kl"} else None
    y_ref = dual_update_torch(xbar, y, sigma, lam_tv)
    x_ref = primal_update_torch(kind, x, y_ref, yobs, tau)
    xb_ref = x_ref + (x_ref - x)
    x_got, y_got, xb_got = pdhg_step(
        kind, x, y, xbar, yobs, tau=tau, sigma=sigma,
        lam_tv=lam_tv, theta=1.0, backend=Backend.TORCH,
    )
    torch.testing.assert_close(y_got, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_got, x_ref, rtol=0, atol=0)
    torch.testing.assert_close(xb_got, xb_ref, rtol=0, atol=0)


def test_no_silent_triton_fallback_on_cpu():
    x, _, obs = tensors(torch.float32)
    with pytest.raises(BackendUnavailableError):
        pdhg("gaussian", x, obs, 0.1, max_iter=2, min_iter=0,
             backend=Backend.TRITON_STENCIL)
    with pytest.raises(BackendUnavailableError):
        pdhg("gaussian", x, obs, 0.1, max_iter=2, min_iter=0,
             backend=Backend.TRITON_ELEMENTWISE)


def test_ill_posed_models_are_rejected():
    _, _, obs = tensors()
    with pytest.raises(ValueError, match="no minimizer"):
        validate_pdhg_model("exp", None)
    with pytest.raises(ValueError, match="unbounded below"):
        validate_pdhg_model("neglog", None)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_pdhg_model("poisson_log", torch.zeros_like(obs))


@pytest.mark.parametrize("kind", [
    "gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx",
])
def test_torch_pdhg_is_deterministic_finite_and_reduces_objective(kind):
    g = torch.Generator().manual_seed(7)
    obs = torch.rand((1,1,24,31), generator=g, dtype=torch.float64) + 0.4
    if kind == "gaussian":
        x0 = obs + 0.25 * torch.randn(obs.shape, generator=g, dtype=obs.dtype)
        yobs = obs
    elif kind == "poisson_log":
        x0 = torch.log(obs)
        yobs = obs
    elif kind in {"poisson_intensity", "kl"}:
        x0 = obs.clone()
        yobs = obs
    else:
        x0 = torch.ones_like(obs)
        yobs = None
    obj0 = float(objective_value(kind, x0, yobs, 0.06))
    out1, dual1, info1 = pdhg(
        kind, x0, yobs, 0.06, max_iter=250, min_iter=30,
        tol=2e-7, backend="torch", monitor_every=10,
    )
    out2, dual2, info2 = pdhg(
        kind, x0, yobs, 0.06, max_iter=250, min_iter=30,
        tol=2e-7, backend="torch", monitor_every=10,
    )
    assert torch.isfinite(out1).all() and torch.isfinite(dual1).all()
    assert info1.backend.executed == "torch"
    assert info1.backend.requested == "torch"
    assert info1.iterations == info2.iterations
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    torch.testing.assert_close(dual1, dual2, rtol=0, atol=0)
    objf = float(objective_value(kind, out1, yobs, 0.06))
    assert objf <= obj0 + 1e-10


def test_noncontiguous_torch_step_matches_contiguous():
    g = torch.Generator().manual_seed(9)
    bigx = torch.randn((1,1,27,35), generator=g, dtype=torch.float64)
    bigy = torch.randn((1,2,27,35), generator=g, dtype=torch.float64)
    bigo = torch.rand((1,1,27,35), generator=g, dtype=torch.float64) + 0.2
    x = bigx[:,:,2:-2,3:-3]
    y = bigy[:,:,2:-2,3:-3]
    obs = bigo[:,:,2:-2,3:-3]
    assert not x.is_contiguous()
    a = pdhg_step("kl", x, y, x, obs, tau=0.2, sigma=0.2,
                  lam_tv=0.08, backend="torch")
    b = pdhg_step("kl", x.contiguous(), y.contiguous(), x.contiguous(),
                  obs.contiguous(), tau=0.2, sigma=0.2,
                  lam_tv=0.08, backend="torch")
    for aa, bb in zip(a,b):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)
