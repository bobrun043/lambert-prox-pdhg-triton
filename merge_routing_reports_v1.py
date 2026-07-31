"""Merge multiple device-local routing reports into one multi-GPU table.

Put files matching ``pdhg_routing_benchmark_raw_v1*.json`` in this directory,
then run this script. Only PASS reports with the expected schema are accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pdhg_routing_table_multi_gpu_v1.json"


def device_key(env: dict) -> str:
    return "|".join([
        env.get("device_name", "unknown"),
        env.get("compute_capability", "unknown"),
        env.get("cuda_runtime", "unknown"),
        env.get("torch", "unknown"),
    ])


def main() -> int:
    paths = sorted(HERE.glob("pdhg_routing_benchmark_raw_v1*.json"))
    devices = {}
    accepted = []
    rejected = []
    policy = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "lambert-prox-pdhg-routing-benchmark-v1" or data.get("status") != "PASS":
            rejected.append({"file": path.name, "reason": "not a PASS routing benchmark"})
            continue
        key = device_key(data["environment"])
        routes = {}
        for case in data["cases"]:
            routes.setdefault(case["kind"], []).append({
                "height": case["height"], "width": case["width"], "pixels": case["pixels"],
                "backend": case["routing_decision"]["selected_backend"],
                "decision": case["routing_decision"],
                "median_seconds": {k: v["timing"]["median_seconds"] for k, v in case["backends"].items()},
                "incremental_peak_allocated_bytes": {
                    k: v["memory"]["incremental_peak_allocated_bytes"] for k, v in case["backends"].items()
                },
            })
        for values in routes.values():
            values.sort(key=lambda x: x["pixels"])
        devices[key] = {
            "environment": data["environment"],
            "provenance": data["provenance"],
            "routes": routes,
            "source_report": path.name,
        }
        accepted.append(path.name)
        policy = {
            "fallback": "torch",
            "minimum_robust_speedup": data["protocol"]["minimum_robust_speedup"],
            "interpolation": "nearest tested pixel count; fallback if area ratio > 4",
        }
    result = {
        "schema": "lambert-prox-pdhg-routing-table-v1",
        "status": "PASS" if devices else "EMPTY",
        "policy": policy or {"fallback": "torch"},
        "devices": devices,
        "merge": {"accepted": accepted, "rejected": rejected},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["merge"], indent=2))
    print(f"wrote {OUT}")
    return 0 if devices else 2


if __name__ == "__main__":
    raise SystemExit(main())
