"""Runtime consumer for a Lambert-Prox PDHG routing table.

The policy is intentionally conservative. It requires an exact environment key
and uses the nearest tested pixel count only when the area ratio is at most 4.
Otherwise it returns Torch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class RouteDecision:
    backend: str
    reason: str
    matched_height: int | None = None
    matched_width: int | None = None
    area_ratio: float | None = None


def current_device_key() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    return "|".join([
        torch.cuda.get_device_name(0),
        f"{props.major}.{props.minor}",
        str(torch.version.cuda),
        str(torch.__version__),
    ])


def select_backend(kind: str, height: int, width: int,
                   table_path: str | Path = "pdhg_routing_table_v1.json",
                   max_area_ratio: float = 4.0) -> RouteDecision:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    path = Path(table_path)
    if not path.exists():
        return RouteDecision("torch", f"routing table not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "lambert-prox-pdhg-routing-table-v1" or data.get("status") != "PASS":
        return RouteDecision("torch", "routing table is not a validated PASS table")
    key = current_device_key()
    device = data.get("devices", {}).get(key)
    if device is None:
        return RouteDecision("torch", "no exact environment entry")
    entries = device.get("routes", {}).get(kind.lower().strip(), [])
    if not entries:
        return RouteDecision("torch", f"no route for model {kind!r}")
    pixels = height * width
    nearest = min(entries, key=lambda e: abs(__import__("math").log(max(e["pixels"], 1) / pixels)))
    ratio = max(nearest["pixels"] / pixels, pixels / nearest["pixels"])
    if ratio > max_area_ratio:
        return RouteDecision("torch", f"nearest tested area is too distant ({ratio:.2f}x)")
    return RouteDecision(
        nearest["backend"],
        "nearest validated size on exact software/GPU environment",
        int(nearest["height"]), int(nearest["width"]), float(ratio),
    )


__all__ = ["RouteDecision", "current_device_key", "select_backend"]
