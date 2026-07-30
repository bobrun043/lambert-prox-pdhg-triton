"""Auto-routed canonical PDHG with explicit, evidence-backed fallback reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import pdhg_promoted_v2 as promoted
from routing_policy_v2 import (
    DEFAULT_EVIDENCE_MANIFEST,
    DEFAULT_TABLE,
    RouteDecision,
    RoutingEvidenceError,
    select_backend,
)


@dataclass(frozen=True)
class AutoRoutingReport:
    mode: str
    selected_backend: str
    executed_backend: str
    fallback_used: bool
    fallback_reason: str | None
    decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pdhg_auto(
    kind: str,
    x0: torch.Tensor,
    y_obs: torch.Tensor | None,
    lam_tv: float,
    *,
    backend: str = "auto",
    routing_table: str | Path = DEFAULT_TABLE,
    routing_evidence_manifest: str | Path = DEFAULT_EVIDENCE_MANIFEST,
    max_area_ratio: float | None = None,
    max_aspect_ratio_factor: float = 2.0,
    safe_auto_fallback: bool = True,
    **pdhg_kwargs: Any,
):
    """Run canonical PDHG with either an explicit backend or conservative auto routing.

    Explicit Triton requests never fallback.  In ``auto`` mode only, unavailable
    or invalid accelerated evidence may fall back to Torch when
    ``safe_auto_fallback=True``; the returned report records that fallback.
    """
    mode = str(backend).lower().strip()
    if mode != "auto":
        selected = promoted.Backend(mode)
        x, y, info = promoted.pdhg(kind, x0, y_obs, lam_tv, backend=selected, **pdhg_kwargs)
        decision = RouteDecision(
            requested=mode, selected_backend=mode, reason="explicit backend request",
            environment_key="explicit", kind=kind.lower().strip(),
            height=int(x0.shape[-2]), width=int(x0.shape[-1]),
        )
        report = AutoRoutingReport(mode, mode, info.backend.executed, False, None, decision.to_dict())
        return x, y, info, report

    try:
        decision = select_backend(
            kind, int(x0.shape[-2]), int(x0.shape[-1]),
            device=x0.device, dtype=x0.dtype,
            table_path=routing_table,
            evidence_manifest=routing_evidence_manifest,
            max_area_ratio=max_area_ratio,
            max_aspect_ratio_factor=max_aspect_ratio_factor,
        )
    except RoutingEvidenceError as exc:
        if not safe_auto_fallback:
            raise
        decision = RouteDecision(
            requested="auto", selected_backend="torch",
            reason=f"routing evidence rejected: {exc}", environment_key="unverified",
            kind=kind.lower().strip(), height=int(x0.shape[-2]), width=int(x0.shape[-1]),
            evidence_verified=False,
        )
    chosen = decision.selected_backend
    try:
        x, y, info = promoted.pdhg(kind, x0, y_obs, lam_tv, backend=chosen, **pdhg_kwargs)
        report = AutoRoutingReport("auto", chosen, info.backend.executed, False, None, decision.to_dict())
        return x, y, info, report
    except (promoted.BackendUnavailableError, promoted.PromotionEvidenceError, ImportError) as exc:
        if chosen == "torch" or not safe_auto_fallback:
            raise
        x, y, info = promoted.pdhg(kind, x0, y_obs, lam_tv, backend="torch", **pdhg_kwargs)
        report = AutoRoutingReport(
            "auto", chosen, "torch", True,
            f"accelerated backend rejected at execution: {exc}", decision.to_dict(),
        )
        return x, y, info, report


__all__ = ["AutoRoutingReport", "pdhg_auto"]
