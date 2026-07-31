import math

import pytest
import torch

from pdhg_close1 import (
    diagnose_solution,
    fidelity_conjugate_value,
    fidelity_value_closed,
    norm_k_periodic_2d,
    norm_k_upper_bound_periodic_2d,
    pdhg,
)


def test_finite_grid_norm_even_and_odd():
    assert norm_k_periodic_2d(32, 48) == pytest.approx(math.sqrt(8.0), rel=0, abs=1e-15)
    odd = norm_k_periodic_2d(31, 47)
    expected = math.sqrt(
        4 * math.sin(math.pi * 23 / 47) ** 2
        + 4 * math.sin(math.pi * 15 / 31) ** 2
    )
    assert odd == pytest.approx(expected, rel=0, abs=1e-15)
    assert odd < norm_k_upper_bound_periodic_2d()


@pytest.mark.parametrize("kind", [
    "gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx",
])
def test_fenchel_young_equality_at_gradient(kind):
    dtype = torch.float64
    x = torch.tensor([[[[0.7, 1.3], [2.0, 0.4]]]], dtype=dtype)
    obs = torch.tensor([[[[0.8, 1.1], [1.4, 0.6]]]], dtype=dtype)
    if kind == "gaussian":
        s, y = x - obs, obs
    elif kind == "poisson_intensity":
        s, y = 1.0 - obs / x, obs
    elif kind == "poisson_log":
        s, y = torch.exp(x) - obs, obs
    elif kind == "kl":
        s, y = torch.log(x / obs), obs
    else:
        s, y = torch.log(x) + 1.0, None
    lhs = fidelity_value_closed(kind, x, y) + fidelity_conjugate_value(kind, s, y)
    rhs = (x * s).sum()
    torch.testing.assert_close(lhs, rhs, rtol=2e-14, atol=2e-14)


def test_constant_gaussian_solution_is_certified():
    obs = torch.full((1, 1, 9, 11), 1.25, dtype=torch.float64)
    x, dual, info = pdhg(
        "gaussian", obs.clone(), obs, 0.08,
        max_iter=30, min_iter=2, tol=1e-12, backend="torch", monitor_every=2,
        kkt_tol=1e-12, gap_tol=1e-12,
    )
    assert info.stabilized
    assert info.certified
    assert info.diagnostics.kkt_max_residual <= 1e-12
    assert info.diagnostics.relative_gap <= 1e-12
    assert info.norm_k_exact < math.sqrt(8.0)
    torch.testing.assert_close(x, obs, rtol=0, atol=0)
    torch.testing.assert_close(dual, torch.zeros_like(dual), rtol=0, atol=0)


def test_underestimated_norm_is_rejected():
    obs = torch.ones((1, 1, 8, 10), dtype=torch.float64)
    with pytest.raises(ValueError, match="underestimates"):
        pdhg("gaussian", obs, obs, 0.1, norm_k=2.0, max_iter=1, min_iter=0)


def test_gap_is_nonnegative_up_to_roundoff():
    obs = torch.rand((1, 1, 7, 9), dtype=torch.float64) + 0.2
    x, dual, info = pdhg(
        "gaussian", obs.clone(), obs, 0.05,
        max_iter=80, min_iter=80, tol=0.0, backend="torch",
    )
    d = diagnose_solution(
        "gaussian", x, dual, obs, 0.05, tau=info.tau, sigma=info.sigma
    )
    assert d.finite
    assert d.primal_dual_gap >= -1e-10
