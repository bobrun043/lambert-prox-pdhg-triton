"""Statistical CUDA benchmark and conservative backend routing for Lambert-Prox PDHG.

No command-line arguments are required. Optional environment variables:

    BENCH_SIZES="64x64,128x192,256x256,512x512,1024x1024"
    BENCH_REPEATS=15
    BENCH_WARMUPS=2
    BENCH_TARGET_MS=150
    BENCH_MIN_ITERS=5
    BENCH_MAX_ITERS=300
    BENCH_BOOTSTRAP=4000
    BENCH_MIN_SPEEDUP=1.10
    BENCH_SEED=20260730

The benchmark excludes first-observed JIT calls from measured samples, randomizes
backend order within every round, uses fixed iteration counts, synchronizes CUDA,
and applies a correctness gate before timing.  It writes raw measurements, a
routing table, and a Markdown summary next to this script.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pdhg_promoted_v2 import Backend, objective_value, pdhg, verify_promotion_manifest

RAW_REPORT = HERE / "pdhg_routing_benchmark_raw_v1.json"
ROUTING_TABLE = HERE / "pdhg_routing_table_v1.json"
SUMMARY_MD = HERE / "pdhg_routing_benchmark_summary_v1.md"

BACKENDS = [Backend.TORCH, Backend.TRITON_ELEMENTWISE, Backend.TRITON_STENCIL]
KINDS = ["gaussian", "poisson_intensity", "poisson_log", "kl", "xlogx"]
CORRECTNESS_TOL = 8.0e-5


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def parse_sizes(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        h, w = token.split("x", 1)
        hv, wv = int(h), int(w)
        if hv <= 0 or wv <= 0:
            raise ValueError(f"invalid size: {token}")
        out.append((hv, wv))
    if not out:
        raise ValueError("BENCH_SIZES produced no sizes")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nvidia_smi_fields() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=uuid,driver_version,power.limit,clocks.max.sm,clocks.max.memory",
        "--format=csv,noheader,nounits",
        "-i", "0",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=True)
        vals = [x.strip() for x in proc.stdout.strip().split(",")]
        if len(vals) == 5:
            return {
                "gpu_uuid": vals[0],
                "driver_version": vals[1],
                "power_limit_w": float(vals[2]),
                "max_sm_clock_mhz": float(vals[3]),
                "max_memory_clock_mhz": float(vals[4]),
            }
    except Exception as exc:
        return {"nvidia_smi_error": str(exc)}
    return {"nvidia_smi_error": "unexpected nvidia-smi output"}


def device_environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
        "torch_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    try:
        import triton
        env["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        env["triton_error"] = str(exc)
    env.update(nvidia_smi_fields())
    return env


def scaled_max(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs()))
    return float(((a - b).abs() / den).max().item())


def make_case(kind: str, h: int, w: int, device: torch.device, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (1, 1, h, w)
    if kind == "gaussian":
        y = torch.rand(shape, generator=g, device=device, dtype=torch.float32) + 0.1
        return y.clone(), y, 0.08
    if kind == "poisson_intensity":
        y = torch.rand(shape, generator=g, device=device, dtype=torch.float32) * 4.0 + 0.05
        return y.clone(), y, 0.04
    if kind == "poisson_log":
        y = torch.rand(shape, generator=g, device=device, dtype=torch.float32) * 3.0 + 0.1
        return torch.log(y), y, 0.04
    if kind == "kl":
        y = torch.rand(shape, generator=g, device=device, dtype=torch.float32) + 0.1
        return y.clone(), y, 0.03
    if kind == "xlogx":
        x0 = torch.rand(shape, generator=g, device=device, dtype=torch.float32) + 0.3
        return x0, None, 0.01
    raise ValueError(kind)


def run_solver(kind: str, x0: torch.Tensor, y_obs: torch.Tensor | None, lam_tv: float,
               iterations: int, backend: Backend):
    return pdhg(
        kind=kind,
        x0=x0,
        y_obs=y_obs,
        lam_tv=lam_tv,
        max_iter=iterations,
        min_iter=iterations,
        tol=0.0,
        backend=backend,
        monitor_every=max(1, iterations),
        promotion_manifest=HERE / "promotion_manifest_v1.json",
    )


def timed_solver(kind: str, x0: torch.Tensor, y_obs: torch.Tensor | None,
                 lam_tv: float, iterations: int, backend: Backend):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = run_solver(kind, x0, y_obs, lam_tv, iterations, backend)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def observed_first_call(kind: str, x0: torch.Tensor, y_obs: torch.Tensor | None,
                        lam_tv: float, backend: Backend) -> float:
    (_, _, _), elapsed = timed_solver(kind, x0, y_obs, lam_tv, 2, backend)
    return elapsed


def measure_peak_memory(kind: str, x0: torch.Tensor, y_obs: torch.Tensor | None,
                        lam_tv: float, iterations: int, backend: Backend) -> dict[str, int]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_alloc = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    out = run_solver(kind, x0, y_obs, lam_tv, iterations, backend)
    torch.cuda.synchronize()
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del out
    return {
        "baseline_allocated_bytes": baseline_alloc,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_alloc - baseline_alloc),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - baseline_reserved),
    }


def summarize_times(times: list[float], pixels: int, iterations: int) -> dict[str, Any]:
    arr = np.asarray(times, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return {
        "samples_seconds": [float(x) for x in times],
        "count": int(arr.size),
        "median_seconds": med,
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "mad_seconds": mad,
        "p10_seconds": float(np.percentile(arr, 10)),
        "p25_seconds": float(np.percentile(arr, 25)),
        "p75_seconds": float(np.percentile(arr, 75)),
        "p90_seconds": float(np.percentile(arr, 90)),
        "median_us_per_iteration": med * 1e6 / iterations,
        "median_ns_per_pixel_iteration": med * 1e9 / (pixels * iterations),
    }


def bootstrap_speedup_ci(torch_times: list[float], candidate_times: list[float],
                         seed: int, draws: int) -> dict[str, float]:
    a = np.asarray(torch_times, dtype=np.float64)
    b = np.asarray(candidate_times, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired timing arrays are required")
    ratios = a / b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(draws, a.size))
    boots = np.median(ratios[idx], axis=1)
    return {
        "paired_median_speedup": float(np.median(ratios)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
    }


def routing_decision(backends: dict[str, Any], correctness: dict[str, Any],
                     seed: int, draws: int, min_speedup: float) -> dict[str, Any]:
    torch_key = Backend.TORCH.value
    torch_times = backends[torch_key]["timing"]["samples_seconds"]
    candidates = []
    for key in (Backend.TRITON_ELEMENTWISE.value, Backend.TRITON_STENCIL.value):
        if not correctness[key]["pass"]:
            continue
        ci = bootstrap_speedup_ci(torch_times, backends[key]["timing"]["samples_seconds"], seed, draws)
        candidates.append((key, backends[key]["timing"]["median_seconds"], ci))
    candidates.sort(key=lambda x: x[1])
    if not candidates:
        return {"selected_backend": torch_key, "reason": "no accelerated backend passed correctness"}
    fastest_key, fastest_med, fastest_ci = candidates[0]
    torch_med = backends[torch_key]["timing"]["median_seconds"]
    if fastest_med >= torch_med:
        return {
            "selected_backend": torch_key,
            "reason": "Torch has the lowest median latency",
            "fastest_accelerated_backend": fastest_key,
            "fastest_accelerated_speedup": fastest_ci,
        }
    # Tie-break two accelerated paths within 3%: prefer lower peak memory, then elementwise.
    if len(candidates) > 1:
        second_key, second_med, _ = candidates[1]
        if second_med / fastest_med <= 1.03:
            mem_fast = backends[fastest_key]["memory"]["incremental_peak_allocated_bytes"]
            mem_second = backends[second_key]["memory"]["incremental_peak_allocated_bytes"]
            if mem_second < mem_fast or (mem_second == mem_fast and second_key == Backend.TRITON_ELEMENTWISE.value):
                fastest_key, fastest_med = second_key, second_med
                fastest_ci = bootstrap_speedup_ci(
                    torch_times, backends[fastest_key]["timing"]["samples_seconds"], seed + 17, draws
                )
    if fastest_ci["ci95_low"] < min_speedup:
        return {
            "selected_backend": torch_key,
            "reason": f"accelerated gain is not robustly above {min_speedup:.2f}x",
            "candidate_backend": fastest_key,
            "candidate_speedup": fastest_ci,
        }
    return {
        "selected_backend": fastest_key,
        "reason": f"paired bootstrap lower 95% bound >= {min_speedup:.2f}x",
        "speedup_vs_torch": fastest_ci,
        "median_seconds": fastest_med,
        "torch_median_seconds": torch_med,
    }


def build_routing_table(report: dict[str, Any]) -> dict[str, Any]:
    env = report["environment"]
    device_key = "|".join([
        env.get("device_name", "unknown"),
        env.get("compute_capability", "unknown"),
        env.get("cuda_runtime", "unknown"),
        env.get("torch", "unknown"),
    ])
    routes: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    for case in report["cases"]:
        routes[case["kind"]].append({
            "height": case["height"],
            "width": case["width"],
            "pixels": case["pixels"],
            "backend": case["routing_decision"]["selected_backend"],
            "decision": case["routing_decision"],
            "median_seconds": {
                k: v["timing"]["median_seconds"] for k, v in case["backends"].items()
            },
            "incremental_peak_allocated_bytes": {
                k: v["memory"]["incremental_peak_allocated_bytes"] for k, v in case["backends"].items()
            },
        })
    for vals in routes.values():
        vals.sort(key=lambda x: x["pixels"])
    return {
        "schema": "lambert-prox-pdhg-routing-table-v1",
        "status": report["status"],
        "policy": {
            "fallback": Backend.TORCH.value,
            "minimum_robust_speedup": report["protocol"]["minimum_robust_speedup"],
            "interpolation": "nearest tested pixel count; fallback if area ratio > 4",
        },
        "devices": {
            device_key: {
                "environment": env,
                "provenance": report["provenance"],
                "routes": routes,
            }
        },
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Lambert-Prox PDHG — benchmark statistique et routage V1",
        "",
        f"Statut : **{report['status']}**",
        "",
        "Ce document décrit des mesures sur l'environnement indiqué. Il ne constitue pas un claim SOTA ni une garantie inter-GPU.",
        "",
        "## Environnement",
        "",
        f"- GPU : `{report['environment'].get('device_name')}`",
        f"- Compute capability : `{report['environment'].get('compute_capability')}`",
        f"- PyTorch : `{report['environment'].get('torch')}`",
        f"- CUDA : `{report['environment'].get('cuda_runtime')}`",
        f"- Triton : `{report['environment'].get('triton', 'unknown')}`",
        "",
        "## Décisions",
        "",
        "| Modèle | Taille | Torch ms | Élémentaire ms | Stencil ms | Route |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for c in report["cases"]:
        b = c["backends"]
        lines.append(
            f"| {c['kind']} | {c['height']}×{c['width']} | "
            f"{1000*b['torch']['timing']['median_seconds']:.3f} | "
            f"{1000*b['triton_elementwise']['timing']['median_seconds']:.3f} | "
            f"{1000*b['triton_stencil']['timing']['median_seconds']:.3f} | "
            f"`{c['routing_decision']['selected_backend']}` |"
        )
    lines += [
        "",
        "## Règle de promotion performance",
        "",
        f"Un backend accéléré est retenu seulement si la borne basse bootstrap à 95 % du gain apparié est au moins `{report['protocol']['minimum_robust_speedup']:.2f}×`. Sinon, la route est `torch`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    sizes = parse_sizes(os.getenv("BENCH_SIZES", "64x64,128x192,256x256,512x512,1024x1024"))
    repeats = env_int("BENCH_REPEATS", 15)
    warmups = env_int("BENCH_WARMUPS", 2)
    target_ms = env_float("BENCH_TARGET_MS", 150.0)
    min_iters = env_int("BENCH_MIN_ITERS", 5)
    max_iters = env_int("BENCH_MAX_ITERS", 300)
    bootstrap_draws = env_int("BENCH_BOOTSTRAP", 4000)
    min_speedup = env_float("BENCH_MIN_SPEEDUP", 1.10)
    seed = env_int("BENCH_SEED", 20260730)
    if repeats < 7:
        raise ValueError("BENCH_REPEATS must be >= 7")
    if warmups < 1:
        raise ValueError("BENCH_WARMUPS must be >= 1")
    if min_iters < 1 or max_iters < min_iters:
        raise ValueError("invalid iteration bounds")

    provenance_files = [
        "promotion_manifest_v1.json",
        "triton_validation_metrics_v2.json",
        "lambert_prox_torch_v1.py",
        "triton_prox_canonical_v1.py",
        "triton_stencil_canonical_v1.py",
        "pdhg_promoted_v2.py",
        "benchmark_promoted_routing_v1.py",
    ]
    provenance = {name: sha256_file(HERE / name) for name in provenance_files}

    if not torch.cuda.is_available():
        report = {
            "schema": "lambert-prox-pdhg-routing-benchmark-v1",
            "status": "NOT_RUN",
            "reason": "CUDA unavailable",
            "provenance": provenance,
            "protocol": {},
            "cases": [],
        }
        RAW_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 3
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Triton is required: {exc}") from exc

    verify_promotion_manifest(HERE / "promotion_manifest_v1.json")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    device = torch.device("cuda")
    cases: list[dict[str, Any]] = []

    for size_idx, (h, w) in enumerate(sizes):
        for kind_idx, kind in enumerate(KINDS):
            case_seed = seed + size_idx * 1000 + kind_idx * 37
            x0, y_obs, lam_tv = make_case(kind, h, w, device, case_seed)
            print(f"\n[{kind} {h}x{w}] compile/correctness/calibration")
            first_calls: dict[str, float] = {}
            correctness: dict[str, Any] = {}
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            for backend in BACKENDS:
                first_calls[backend.value] = observed_first_call(kind, x0, y_obs, lam_tv, backend)
                x, y, _ = run_solver(kind, x0, y_obs, lam_tv, 20, backend)
                outputs[backend.value] = (x, y)
            x_ref, y_ref = outputs[Backend.TORCH.value]
            obj_ref = float(objective_value(kind, x_ref, y_obs, lam_tv).item())
            for backend in BACKENDS:
                x, y = outputs[backend.value]
                ps = scaled_max(x, x_ref)
                ds = scaled_max(y, y_ref)
                obj = float(objective_value(kind, x, y_obs, lam_tv).item())
                correctness[backend.value] = {
                    "primal_scaled": ps,
                    "dual_scaled": ds,
                    "objective": obj,
                    "objective_abs_delta_vs_torch": abs(obj - obj_ref),
                    "pass": max(ps, ds) <= CORRECTNESS_TOL,
                }
            if not all(v["pass"] for v in correctness.values()):
                raise RuntimeError(f"correctness gate failed for {kind} {h}x{w}: {correctness}")

            (_, _, _), calib_seconds = timed_solver(kind, x0, y_obs, lam_tv, 10, Backend.TORCH)
            per_iter = max(calib_seconds / 10.0, 1e-9)
            iterations = int(round((target_ms / 1000.0) / per_iter))
            iterations = max(min_iters, min(max_iters, iterations))
            print(f"  calibrated iterations/sample={iterations}")

            for backend in BACKENDS:
                for _ in range(warmups):
                    run_solver(kind, x0, y_obs, lam_tv, iterations, backend)
                torch.cuda.synchronize()

            times: dict[str, list[float]] = {b.value: [] for b in BACKENDS}
            rng = random.Random(case_seed + 991)
            for round_idx in range(repeats):
                order = BACKENDS.copy()
                rng.shuffle(order)
                for backend in order:
                    (_, _, _), elapsed = timed_solver(kind, x0, y_obs, lam_tv, iterations, backend)
                    times[backend.value].append(elapsed)
                print(f"  round {round_idx + 1}/{repeats}", end="\r", flush=True)
            print()

            backends: dict[str, Any] = {}
            pixels = h * w
            for backend in BACKENDS:
                mem = measure_peak_memory(kind, x0, y_obs, lam_tv, iterations, backend)
                backends[backend.value] = {
                    "first_observed_call_seconds": first_calls[backend.value],
                    "timing": summarize_times(times[backend.value], pixels, iterations),
                    "memory": mem,
                }
            decision = routing_decision(
                backends, correctness, case_seed + 12345, bootstrap_draws, min_speedup
            )
            case = {
                "kind": kind,
                "height": h,
                "width": w,
                "pixels": pixels,
                "lam_tv": lam_tv,
                "iterations_per_sample": iterations,
                "correctness": correctness,
                "backends": backends,
                "routing_decision": decision,
            }
            cases.append(case)
            print(f"  route -> {decision['selected_backend']} ({decision['reason']})")
            # Checkpoint after every case.
            checkpoint = {
                "schema": "lambert-prox-pdhg-routing-benchmark-v1",
                "status": "RUNNING",
                "environment": device_environment(),
                "provenance": provenance,
                "protocol": {
                    "sizes": [f"{a}x{b}" for a, b in sizes],
                    "repeats": repeats,
                    "warmups": warmups,
                    "target_sample_ms": target_ms,
                    "min_iterations": min_iters,
                    "max_iterations": max_iters,
                    "bootstrap_draws": bootstrap_draws,
                    "minimum_robust_speedup": min_speedup,
                    "correctness_tolerance": CORRECTNESS_TOL,
                    "timing_scope": "fixed-iteration full pdhg API; first observed JIT calls excluded",
                },
                "cases": cases,
            }
            RAW_REPORT.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    report = {
        "schema": "lambert-prox-pdhg-routing-benchmark-v1",
        "status": "PASS",
        "environment": device_environment(),
        "provenance": provenance,
        "protocol": {
            "sizes": [f"{a}x{b}" for a, b in sizes],
            "repeats": repeats,
            "warmups": warmups,
            "target_sample_ms": target_ms,
            "min_iterations": min_iters,
            "max_iterations": max_iters,
            "bootstrap_draws": bootstrap_draws,
            "minimum_robust_speedup": min_speedup,
            "correctness_tolerance": CORRECTNESS_TOL,
            "timing_scope": "fixed-iteration full pdhg API; first observed JIT calls excluded",
            "backend_order": "randomized independently in every paired round",
            "cuda_synchronization": "before and after every timed sample",
            "claim_scope": "device-local routing only; no SOTA or inter-GPU claim",
        },
        "cases": cases,
    }
    RAW_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = build_routing_table(report)
    ROUTING_TABLE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(markdown_summary(report), encoding="utf-8")
    print(f"\nBENCHMARK: PASS\nraw={RAW_REPORT}\nrouting={ROUTING_TABLE}\nsummary={SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    code = main()
    if code == 0:
        print("ROUTING_BENCHMARK: PASS")
    elif code == 3:
        print("ROUTING_BENCHMARK: NOT_RUN")
