from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import lambert_prox_reference_v1 as ref
import lambert_prox_torch_v1 as tt


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _normalized(a: torch.Tensor, scale: torch.Tensor) -> float:
    return float((torch.abs(a) / torch.maximum(torch.ones_like(scale), scale)).max())


def test_module_is_independent_of_legacy_implementations():
    src = inspect.getsource(tt)
    forbidden = ["solve_u_log_u.py", "torch_lambert_prox_v3", "triton_prox", "pdhg_solver_v5"]
    assert all(name not in src for name in forbidden)


def test_fp64_pair_matches_numpy_oracle_across_extremes():
    R_np = np.unique(
        np.concatenate(
            (
                np.array([-1000.0, -800.0, -745.0, -100.0, -20.0, -8.0001, -8.0, -7.9999]),
                np.linspace(-7.5, 8.5, 10001),
                np.geomspace(10.0, 1e6, 2000),
            )
        )
    )
    oracle = ref.solve_u_log_u_reference(R_np)
    R = torch.from_numpy(R_np)
    out = tt.solve_u_log_u_pair(R)
    assert bool(torch.all(out.converged))

    q_err = np.abs(_np(out.q) - oracle.q)
    assert np.max(q_err / np.maximum(1.0, np.abs(oracle.q))) < 8e-15

    representable = oracle.u > 0.0
    u_err = np.abs(_np(out.u)[representable] - oracle.u[representable])
    assert np.max(u_err / np.maximum(1.0, np.abs(oracle.u[representable]))) < 8e-15
    assert out.status[0].item() == int(tt.SolveStatus.OK_LOG_ONLY)


def test_fp32_contract_against_rounded_oracle():
    R_np = np.unique(
        np.concatenate(
            (
                np.array([-1000.0, -100.0, -20.0, -8.0001, -8.0, -7.9999], dtype=np.float32),
                np.linspace(-7.5, 40.0, 20001, dtype=np.float32),
                np.geomspace(50.0, 1e6, 2000).astype(np.float32),
            )
        )
    ).astype(np.float32)
    oracle = ref.solve_u_log_u_reference(R_np.astype(np.float64))
    out = tt.solve_u_log_u_pair(torch.from_numpy(R_np))
    assert bool(torch.all(out.converged))
    q_ref32 = oracle.q.astype(np.float32)
    q_abs = np.abs(_np(out.q) - q_ref32)
    assert np.max(q_abs / np.maximum(1.0, np.abs(q_ref32))) < 8e-7
    assert float(out.residual.max()) < 2e-4


def test_invalid_status_and_dtype_contract():
    R = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    out = tt.solve_u_log_u_pair(R)
    assert torch.all(out.status == int(tt.SolveStatus.INVALID_INPUT))
    assert torch.isnan(out.u).all()
    with pytest.raises(TypeError):
        tt.solve_u_log_u_pair(torch.tensor([0.0], dtype=torch.float16))
    with pytest.raises(TypeError):
        tt.solve_u_log_u_pair(torch.tensor([0]))
    with pytest.raises(ValueError):
        tt.solve_u_log_u_pair(torch.tensor([0.0]), iters=0)


def test_native_autograd_matches_implicit_first_and_second_derivatives():
    R = torch.linspace(-20.0, 20.0, 2001, dtype=torch.float64, requires_grad=True)
    out = tt.solve_u_log_u_pair(R)

    du = torch.autograd.grad(out.u.sum(), R, create_graph=True)[0]
    du_exact = out.u / (1.0 + out.u)
    assert float((du - du_exact).abs().max().detach()) < 2e-15
    d2u = torch.autograd.grad(du.sum(), R, retain_graph=True)[0]
    d2u_exact = out.u / (1.0 + out.u) ** 3
    assert float((d2u - d2u_exact).abs().max().detach()) < 4e-15

    dq = torch.autograd.grad(out.q.sum(), R, create_graph=True)[0]
    dq_exact = 1.0 / (1.0 + out.u)
    assert float((dq - dq_exact).abs().max().detach()) < 2e-15
    d2q = torch.autograd.grad(dq.sum(), R)[0]
    d2q_exact = -out.u / (1.0 + out.u) ** 3
    assert float((d2q - d2q_exact).abs().max().detach()) < 4e-15


def test_gradcheck_and_gradgradcheck_primitive():
    R = torch.tensor([-6.0, -1.0, 0.5, 5.0, 20.0], dtype=torch.float64, requires_grad=True)
    fn_u = lambda z: tt.solve_u_log_u_pair(z).u
    fn_q = lambda z: tt.solve_u_log_u_pair(z).q
    assert gradcheck(fn_u, (R,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert gradgradcheck(fn_u, (R,), eps=1e-6, atol=2e-6, rtol=2e-5)
    assert gradcheck(fn_q, (R,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert gradgradcheck(fn_q, (R,), eps=1e-6, atol=2e-6, rtol=2e-5)


def test_all_prox_match_numpy_reference_fp64():
    rng = np.random.default_rng(1234)
    n = 12000
    v_np = rng.normal(0.0, 6.0, n)
    lam_np = np.exp(rng.uniform(-5.0, 5.0, n))
    y_np = np.exp(rng.uniform(-8.0, 8.0, n))
    yg_np = rng.normal(size=n)
    v = torch.from_numpy(v_np)
    lam = torch.from_numpy(lam_np)
    y = torch.from_numpy(y_np)
    yg = torch.from_numpy(yg_np)

    cases = [
        (tt.prox_exp(v, lam), ref.prox_exp(v_np, lam_np)),
        (tt.prox_xlogx(v, lam), ref.prox_xlogx(v_np, lam_np)),
        (tt.prox_kl(v, y, lam), ref.prox_kl(v_np, y_np, lam_np)),
        (tt.prox_poisson_log(v, y, lam), ref.prox_poisson_log(v_np, y_np, lam_np)),
        (tt.prox_poisson_intensity(v, y, lam), ref.prox_poisson_intensity(v_np, y_np, lam_np)),
        (tt.prox_neglog(v, lam), ref.prox_neglog(v_np, lam_np)),
        (tt.prox_gaussian(v, yg, lam), ref.prox_gaussian(v_np, yg_np, lam_np)),
    ]
    for got, expected in cases:
        delta = np.abs(_np(got) - expected)
        assert np.max(delta / np.maximum(1.0, np.abs(expected))) < 2e-13


def test_all_prox_kkt_fp64_and_fp32():
    gen = torch.Generator().manual_seed(42)
    for dtype, threshold in [(torch.float64, 2e-13), (torch.float32, 4e-5)]:
        n = 20000
        v = torch.randn(n, dtype=dtype, generator=gen) * 5.0
        lam = torch.exp(torch.empty(n, dtype=dtype).uniform_(-4.0, 4.0, generator=gen))
        y = torch.exp(torch.empty(n, dtype=dtype).uniform_(-5.0, 5.0, generator=gen))

        x = tt.prox_exp(v, lam)
        assert _normalized(x - v + lam * torch.exp(x), torch.abs(v) + torch.abs(x) + lam * torch.exp(x)) < threshold

        x = tt.prox_xlogx(v, lam)
        log_x = tt.prox_xlogx_log(v, lam)
        term = lam * (1.0 + log_x)
        assert _normalized(x - v + term, torch.abs(v) + torch.abs(x) + torch.abs(term)) < threshold

        x = tt.prox_kl(v, y, lam)
        log_x = tt.prox_kl_log(v, y, lam)
        term = lam * (log_x - torch.log(y))
        assert _normalized(x - v + term, torch.abs(v) + torch.abs(x) + torch.abs(term)) < threshold

        x = tt.prox_poisson_log(v, y, lam)
        term = lam * (torch.exp(x) - y)
        assert _normalized(x - v + term, torch.abs(v) + torch.abs(x) + lam * (torch.exp(x) + y)) < threshold

        x = tt.prox_poisson_intensity(v, y, lam)
        term = lam * (1.0 - y / x)
        assert _normalized(x - v + term, torch.abs(v) + torch.abs(x) + lam * (1.0 + y / x)) < threshold

        x = tt.prox_neglog(v, lam)
        assert _normalized(x - v - lam / x, torch.abs(v) + torch.abs(x) + lam / x) < threshold

        yg = torch.randn(n, dtype=dtype, generator=gen)
        x = tt.prox_gaussian(v, yg, lam)
        assert _normalized(x - v + lam * (x - yg), torch.abs(v) + torch.abs(x) + lam * torch.abs(x - yg)) < threshold


def test_stable_extreme_quadratic_roots():
    v = torch.tensor([-1e2, -1e8, -1e150], dtype=torch.float64)
    y = torch.tensor([1e-10, 1e-20, 1e-100], dtype=torch.float64)
    lam = torch.ones_like(v)
    x = tt.prox_poisson_intensity(v, y, lam)
    assert torch.all(x > 0.0)
    resid = x - v + lam * (1.0 - y / x)
    assert _normalized(resid, torch.abs(v) + torch.abs(x) + lam * (1.0 + y / x)) < 5e-16
    x = tt.prox_neglog(v, lam)
    assert torch.all(x > 0.0)
    resid = x - v - lam / x
    assert _normalized(resid, torch.abs(v) + torch.abs(x) + lam / x) < 5e-16


def test_kl_zero_boundary_and_domain_errors():
    v = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(v)
    x = tt.prox_kl(v, y, 0.7)
    log_x = tt.prox_kl_log(v, y, 0.7)
    assert torch.equal(x, torch.zeros_like(x))
    assert torch.isneginf(log_x).all()
    g = torch.autograd.grad(x.sum(), v)[0]
    assert torch.equal(g, torch.zeros_like(g))
    with pytest.raises(ValueError):
        tt.prox_kl(v.detach(), -torch.ones_like(v), 0.7)
    with pytest.raises(ValueError):
        tt.prox_exp(v.detach(), 0.0)


def test_prox_gradcheck_and_gradgradcheck_wrt_v():
    v = torch.tensor([-2.0, -0.2, 0.5, 3.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.2, 0.8, 1.5, 4.0], dtype=torch.float64)
    lam = 0.7
    funcs = [
        lambda z: tt.prox_exp(z, lam),
        lambda z: tt.prox_xlogx(z, lam),
        lambda z: tt.prox_kl(z, y, lam),
        lambda z: tt.prox_poisson_log(z, y, lam),
        lambda z: tt.prox_poisson_intensity(z, y, lam),
        lambda z: tt.prox_neglog(z, lam),
        lambda z: tt.prox_gaussian(z, y, lam),
    ]
    for fn in funcs:
        assert gradcheck(fn, (v,), eps=1e-6, atol=2e-7, rtol=2e-5)
        assert gradgradcheck(fn, (v,), eps=1e-6, atol=3e-6, rtol=3e-5)


def test_broadcast_and_scalar_contract():
    v = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float64)
    lam = torch.tensor([[0.1, 1.0, 10.0]], dtype=torch.float64)
    x = tt.prox_exp(v, lam)
    assert x.shape == (3, 3)
    scalar = tt.solve_u_log_u_pair(torch.tensor(0.0, dtype=torch.float64))
    assert scalar.u.shape == torch.Size([])
    assert bool(scalar.converged)
