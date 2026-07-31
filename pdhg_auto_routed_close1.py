"""Evidence-backed auto routing with CLOSE1 diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import pdhg_close1
import pdhg_promoted_v2 as legacy
from routing_policy_v2 import (
    DEFAULT_EVIDENCE_MANIFEST,
    DEFAULT_TABLE,
    RouteDecision,
    RoutingEvidenceError,
    select_backend,
)


@dataclass(frozen=True)
class AutoRoutingReportClose1:
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
    mode = str(backend).lower().strip()
    if mode != "auto":
        selected = legacy.Backend(mode)
        x, dual, info = pdhg_close1.pdhg(
            kind, x0, y_obs, lam_tv, backend=selected, **pdhg_kwargs
        )
        decision = RouteDecision(
            requested=mode, selected_backend=mode, reason="explicit backend request",
            environment_key="explicit", kind=kind.lower().strip(),
            height=int(x0.shape[-2]), width=int(x0.shape[-1]),
        )
        report = AutoRoutingReportClose1(
            mode, mode, info.backend.executed, False, None, decision.to_dict()
        )
        return x, dual, info, report

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
            kind=kind.lower().strip(), height=int(x0.shape[-2]),
            width=int(x0.shape[-1]), evidence_verified=False,
        )

    chosen = decision.selected_backend
    try:
        x, dual, info = pdhg_close1.pdhg(
            kind, x0, y_obs, lam_tv, backend=chosen, **pdhg_kwargs
        )
        report = AutoRoutingReportClose1(
            "auto", chosen, info.backend.executed, False, None, decision.to_dict()
        )
        return x, dual, info, report
    except (legacy.BackendUnavailableError, legacy.PromotionEvidenceError, ImportError) as exc:
        if chosen == "torch" or not safe_auto_fallback:
            raise
        x, dual, info = pdhg_close1.pdhg(
            kind, x0, y_obs, lam_tv, backend="torch", **pdhg_kwargs
        )
        report = AutoRoutingReportClose1(
            "auto", chosen, "torch", True,
            f"accelerated backend rejected at execution: {exc}", decision.to_dict(),
        )
        return x, dual, info, report


__all__ = ["AutoRoutingReportClose1", "pdhg_auto"]
