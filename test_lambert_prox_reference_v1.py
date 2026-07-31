from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

import lambert_prox_reference_v1 as ref


mp.mp.dps = 100


def test_bicoordinate_contract_extreme_negative():
    R = np.array([-8.0001, -20.0, -100.0, -745.0, -1000.0])
    out = ref.solve_u_log_u_reference(R)
    assert np.all(out.converged)
    assert out.status[-1] == int(ref.SolveStatus.OK_LOG_ONLY)
    assert out.u[-1] == 0.0
    assert np.isfinite(out.q[-1])
    assert out.residual.max() <= 2e-13


def test_threshold_neighborhoods_are_continuous():
    points = np.array([
        -8.0 - 1e-12, -8.0, -8.0 + 1e-12,
        -0.3 - 1e-12, -0.3, -0.3 + 1e-12,
        8.0 - 1e-12, 8.0, 8.0 + 1e-12,
    ])
    out = ref.solve_u_log_u_reference(points)
    assert np.all(out.converged)
    assert np.all(np.diff(out.u) >= 0.0)
    assert out.residual.max() < 1e-13


def test_against_mpmath_value_and_log_coordinates():
    R = np.array([-1000.0, -800.0, -100.0, -20.0, -8.1, -8.0, -1.0, 0.0, 8.0, 100.0, 1e6])
    out = ref.solve_u_log_u_reference(R)
    assert np.all(out.converged)
    for i, r in enumerate(R):
        u_mp = mp.lambertw(mp.e ** mp.mpf(str(r)))
        q_mp = mp.log(u_mp)
        if out.u[i] > 0.0:
            rel = abs(out.u[i] - float(u_mp)) / max(1.0, abs(float(u_mp)))
            assert rel < 5e-15
        assert abs(out.q[i] - float(q_mp)) <= 5e-13 * max(1.0, abs(float(q_mp)))


def test_monotonicity_and_exact_derivatives():
    R = np.linspace(-700.0, 700.0, 20001)
    out = ref.solve_u_log_u_reference(R)
    assert np.all(out.converged)
    assert np.all(np.diff(out.q) >= -2e-14)
    assert np.all(np.diff(out.u) >= 0.0)
    du = ref.derivative_u_from_result(out)
    dq = ref.derivative_q_from_result(out)
    assert np.all((du >= 0.0) & (du < 1.0))
    assert np.all((dq > 0.0) & (dq <= 1.0))
    assert np.max(np.abs(du + dq - 1.0)) <= np.finfo(np.float64).eps


def test_invalid_input_status_and_bad_configuration():
    out = ref.solve_u_log_u_reference([np.nan, np.inf, -np.inf])
    assert np.all(out.status == int(ref.SolveStatus.INVALID_INPUT))
    with pytest.raises(ValueError):
        ref.solve_u_log_u_reference(0.0, max_iter=0)
    with pytest.raises(ValueError):
        ref.solve_u_log_u_reference(0.0, atol=-1.0)


def test_prox_kkt_moderate_domain():
    rng = np.random.default_rng(123)
    n = 10000
    v = rng.normal(0.0, 4.0, n)
    lam = np.exp(rng.uniform(-4.0, 4.0, n))
    y = np.exp(rng.uniform(-5.0, 5.0, n))

    x = ref.prox_exp(v, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam * np.exp(x))
    assert np.max(np.abs(x - v + lam * np.exp(x)) / scale) < 2e-14

    x = ref.prox_xlogx(v, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam * np.abs(1.0 + np.log(x)))
    assert np.max(np.abs(x - v + lam * (1.0 + np.log(x))) / scale) < 1e-13

    x = ref.prox_kl(v, y, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam * np.abs(np.log(x / y)))
    assert np.max(np.abs(x - v + lam * np.log(x / y)) / scale) < 1e-13

    x = ref.prox_poisson_log(v, y, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam * (np.exp(x) + y))
    assert np.max(np.abs(x - v + lam * (np.exp(x) - y)) / scale) < 3e-14

    x = ref.prox_poisson_intensity(v, y, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam * (1.0 + y / x))
    assert np.max(np.abs(x - v + lam * (1.0 - y / x)) / scale) < 3e-14

    x = ref.prox_neglog(v, lam)
    scale = np.maximum(1.0, np.abs(v) + np.abs(x) + lam / x)
    assert np.max(np.abs(x - v - lam / x) / scale) < 3e-14

    yg = rng.normal(size=n)
    x = ref.prox_gaussian(v, yg, lam)
    assert np.max(np.abs(x - v + lam * (x - yg))) < 1e-12


def test_kl_y_zero_is_exact_boundary_not_clamp():
    v = np.array([-100.0, 0.0, 100.0])
    y = np.zeros_like(v)
    x = ref.prox_kl(v, y, 0.7)
    assert np.array_equal(x, np.zeros_like(x))
    with pytest.raises(ValueError):
        ref.prox_kl(v, -np.ones_like(v), 0.7)


def test_stable_quadratic_roots_avoid_cancellation():
    v = np.array([-1e2, -1e8, -1e150])
    y = np.array([1e-10, 1e-20, 1e-100])
    lam = np.ones_like(v)
    x = ref.prox_poisson_intensity(v, y, lam)
    assert np.all(x > 0.0)
    normalized = np.abs(x - v + lam * (1.0 - y / x)) / np.maximum(
        1.0, np.abs(v) + np.abs(x) + lam * (1.0 + y / x)
    )
    assert normalized.max() < 5e-16

    xlog = ref.prox_neglog(v, lam)
    assert np.all(xlog > 0.0)
    normalized = np.abs(xlog - v - lam / xlog) / np.maximum(
        1.0, np.abs(v) + np.abs(xlog) + lam / xlog
    )
    assert normalized.max() < 5e-16


def test_no_general_b_negative_solver_is_exposed():
    assert not hasattr(ref, "solve_atb_logt_eq_c")


def test_scalar_shape_and_broadcast_contract():
    out = ref.solve_u_log_u_reference(0.0)
    assert out.u.shape == ()
    assert out.q.shape == ()
    assert bool(out.converged)
    v = np.array([[-2.0], [0.0], [2.0]])
    lam = np.array([[0.1, 1.0, 10.0]])
    x = ref.prox_exp(v, lam)
    assert x.shape == (3, 3)
    assert np.isfinite(x).all()
