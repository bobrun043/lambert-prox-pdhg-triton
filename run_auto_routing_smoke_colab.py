"""Colab smoke test for the production auto router (no argparse)."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from routing_policy_v2 import current_device_key, select_backend, verify_routing_evidence
from pdhg_auto_routed_v3 import pdhg_auto


def main():
    verify_routing_evidence()
    if not torch.cuda.is_available():
        print("AUTO_ROUTING_SMOKE: NOT_RUN (CUDA unavailable)")
        return {"status": "NOT_RUN"}
    device = torch.device("cuda")
    print("environment_key=", current_device_key())
    rows = []
    g = torch.Generator(device=device).manual_seed(123)
    for kind in ["gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx"]:
        shape = (1, 1, 128, 192)
        if kind == "gaussian":
            y = torch.rand(shape, device=device, generator=g) + 0.1; x0 = y.clone(); lam_tv = 0.08
        elif kind == "poisson_intensity":
            y = torch.rand(shape, device=device, generator=g) * 4.0 + 0.1; x0 = y.clone(); lam_tv = 0.04
        elif kind == "poisson_log":
            y = torch.rand(shape, device=device, generator=g) * 4.0 + 0.1; x0 = torch.log(y); lam_tv = 0.04
        elif kind == "kl":
            y = torch.rand(shape, device=device, generator=g) + 0.1; x0 = y.clone(); lam_tv = 0.03
        else:
            y = None; x0 = torch.rand(shape, device=device, generator=g) + 0.5; lam_tv = 0.01
        decision = select_backend(kind, 128, 192, device=device, dtype=torch.float32)
        x, dual, info, route = pdhg_auto(
            kind, x0.float(), None if y is None else y.float(), lam_tv,
            max_iter=10, min_iter=10, tol=0.0, monitor_every=10,
        )
        row = {"kind": kind, "selected": decision.selected_backend, "executed": route.executed_backend,
               "finite": bool(torch.isfinite(x).all() and torch.isfinite(dual).all())}
        assert row["selected"] == "triton_stencil" and row["executed"] == "triton_stencil" and row["finite"]
        print(kind, "PASS", row)
        rows.append(row)
    result = {"status": "PASS", "environment_key": current_device_key(), "rows": rows}
    Path("auto_routing_smoke_v1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("AUTO_ROUTING_SMOKE: PASS")
    return result


if __name__ == "__main__":
    main()
