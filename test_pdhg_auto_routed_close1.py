import torch

from pdhg_auto_routed_close1 import pdhg_auto
import pdhg_promoted_v2 as legacy


def test_cpu_auto_route_is_torch_and_diagnostic():
    obs = torch.full((1, 1, 9, 13), 0.7, dtype=torch.float64)
    x, dual, info, route = pdhg_auto(
        "gaussian", obs.clone(), obs, 0.06,
        max_iter=10, min_iter=2, tol=1e-12, monitor_every=2,
    )
    assert route.executed_backend == "torch"
    assert route.decision["selected_backend"] == "torch"
    assert info.certified and info.stabilized
    assert torch.isfinite(x).all() and torch.isfinite(dual).all()


def test_explicit_triton_never_silently_falls_back_on_cpu():
    obs = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    try:
        pdhg_auto(
            "gaussian", obs, obs, 0.06, backend="triton_stencil",
            max_iter=1, min_iter=0,
        )
    except legacy.BackendUnavailableError:
        return
    raise AssertionError("explicit Triton request silently fell back")
