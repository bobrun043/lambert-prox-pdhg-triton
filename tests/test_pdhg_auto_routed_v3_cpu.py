from __future__ import annotations

import torch

from pdhg_auto_routed_v3 import pdhg_auto
import pdhg_promoted_v2 as promoted


def run() -> int:
    torch.manual_seed(0)
    y = torch.rand((1, 1, 24, 31), dtype=torch.float64) + 0.1
    x0 = y.clone()
    initial = float(promoted.objective_value("gaussian", x0, y, 0.08))
    x, dual, info, route = pdhg_auto(
        "gaussian", x0, y, 0.08, max_iter=30, min_iter=30, tol=0.0, monitor_every=10,
    )
    final = float(promoted.objective_value("gaussian", x, y, 0.08))
    assert route.executed_backend == "torch"
    assert route.decision["selected_backend"] == "torch"
    assert final <= initial + 1e-10
    assert torch.isfinite(x).all() and torch.isfinite(dual).all()
    try:
        pdhg_auto("gaussian", x0, y, 0.08, backend="triton_stencil", max_iter=1, min_iter=0)
    except promoted.BackendUnavailableError:
        pass
    else:
        raise AssertionError("explicit Triton request silently fell back")
    print("PDHG_AUTO_ROUTED_V3_CPU: PASS (5 assertion groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
