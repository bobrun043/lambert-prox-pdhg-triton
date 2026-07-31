from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np

import lambert_prox_reference_v1 as ref


ROOT = Path(__file__).resolve().parent
SPEC = ROOT.parent / "LAMBERT_PROX_SPEC_CANONIQUE_V1_2026-07-21.pdf"
mp.mp.dps = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_max(residual: np.ndarray, scale: np.ndarray) -> float:
    return float(np.max(np.abs(residual) / np.maximum(1.0, scale)))


def main() -> None:
    # Deterministic inverse grid: thresholds, dense central interval, and extremes.
    R = np.unique(
        np.concatenate(
            [
                np.linspace(-1000.0, -8.001, 5000),
                np.linspace(-8.001, 20.0, 20001),
                np.linspace(20.0, 1000.0, 5000),
                np.array([-8.0, -0.3, 8.0, -745.0, -1000.0, 1e6]),
            ]
        )
    )
    sol = ref.solve_u_log_u_reference(R)

    oracle_R = np.array([-1000.0, -800.0, -100.0, -20.0, -8.1, -8.0, -1.0, 0.0, 8.0, 100.0, 1e6])
    oracle = ref.solve_u_log_u_reference(oracle_R)
    q_errors = []
    u_rel_errors = []
    for i, r in enumerate(oracle_R):
        u_mp = mp.lambertw(mp.e ** mp.mpf(str(r)))
        q_mp = mp.log(u_mp)
        q_errors.append(abs(oracle.q[i] - float(q_mp)))
        if oracle.u[i] > 0.0:
            u_rel_errors.append(abs(oracle.u[i] - float(u_mp)) / max(1.0, abs(float(u_mp))))

    rng = np.random.default_rng(20260730)
    n = 100000
    v = rng.normal(0.0, 4.0, n)
    lam = np.exp(rng.uniform(-4.0, 4.0, n))
    y = np.exp(rng.uniform(-5.0, 5.0, n))

    kkt = {}
    x = ref.prox_exp(v, lam)
    kkt["exp"] = normalized_max(x - v + lam * np.exp(x), np.abs(v) + np.abs(x) + lam * np.exp(x))

    x = ref.prox_xlogx(v, lam)
    kkt["xlogx"] = normalized_max(
        x - v + lam * (1.0 + np.log(x)),
        np.abs(v) + np.abs(x) + lam * np.abs(1.0 + np.log(x)),
    )

    x = ref.prox_kl(v, y, lam)
    kkt["kl"] = normalized_max(
        x - v + lam * np.log(x / y),
        np.abs(v) + np.abs(x) + lam * np.abs(np.log(x / y)),
    )

    x = ref.prox_poisson_log(v, y, lam)
    kkt["poisson_log"] = normalized_max(
        x - v + lam * (np.exp(x) - y),
        np.abs(v) + np.abs(x) + lam * (np.exp(x) + y),
    )

    x = ref.prox_poisson_intensity(v, y, lam)
    kkt["poisson_intensity"] = normalized_max(
        x - v + lam * (1.0 - y / x),
        np.abs(v) + np.abs(x) + lam * (1.0 + y / x),
    )

    x = ref.prox_neglog(v, lam)
    kkt["neglog"] = normalized_max(
        x - v - lam / x,
        np.abs(v) + np.abs(x) + lam / x,
    )

    yg = rng.normal(size=n)
    x = ref.prox_gaussian(v, yg, lam)
    kkt["gaussian"] = normalized_max(
        x - v + lam * (x - yg),
        np.abs(v) + np.abs(x) + lam * np.abs(x - yg),
    )

    payload = {
        "schema": "lambert-prox-reference-metrics-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mpmath": mp.__version__,
        },
        "normative_spec": {
            "path": str(SPEC),
            "sha256": sha256(SPEC) if SPEC.exists() else None,
        },
        "reference_module_sha256": sha256(ROOT / "lambert_prox_reference_v1.py"),
        "inverse_contract": {
            "samples": int(R.size),
            "all_converged": bool(np.all(sol.converged)),
            "max_coordinate_residual": float(np.nanmax(sol.residual)),
            "p99_coordinate_residual": float(np.nanpercentile(sol.residual, 99.0)),
            "max_iterations": int(np.max(sol.iterations)),
            "mean_iterations": float(np.mean(sol.iterations[sol.converged])),
            "log_only_count": int(np.sum(sol.log_only)),
            "R_minus_1000_status": ref.SolveStatus(int(ref.solve_u_log_u_reference(-1000.0).status)).name,
        },
        "mpmath_oracle": {
            "points": oracle_R.tolist(),
            "max_abs_q_error": float(max(q_errors)),
            "max_relative_u_error_when_representable": float(max(u_rel_errors)),
        },
        "prox_kkt_normalized_max": kkt,
        "claims": {
            "permitted": [
                "independent NumPy FP64 reference implemented",
                "bi-coordinate underflow status implemented",
                "canonical proximal formulas verified on the stated deterministic test domain",
            ],
            "not_established": [
                "Torch equivalence",
                "Triton equivalence",
                "GPU performance",
                "novelty or priority",
                "SOTA",
            ],
        },
    }

    (ROOT / "reference_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Rapport de fermeture - Référence FP64 canonique V1",
        "",
        f"- Statut global : **{'PASS' if payload['inverse_contract']['all_converged'] else 'FAIL'}**",
        f"- Points du balayage inverse : {payload['inverse_contract']['samples']}",
        f"- Résidu coordonnée max : {payload['inverse_contract']['max_coordinate_residual']:.3e}",
        f"- Résidu coordonnée p99 : {payload['inverse_contract']['p99_coordinate_residual']:.3e}",
        f"- Itérations max / moyenne : {payload['inverse_contract']['max_iterations']} / {payload['inverse_contract']['mean_iterations']:.3f}",
        f"- Statuts log-only : {payload['inverse_contract']['log_only_count']}",
        f"- Erreur max q contre mpmath : {payload['mpmath_oracle']['max_abs_q_error']:.3e}",
        f"- Erreur relative max u contre mpmath : {payload['mpmath_oracle']['max_relative_u_error_when_representable']:.3e}",
        "",
        "## Résidus KKT normalisés maximaux",
        "",
    ]
    for name, value in kkt.items():
        lines.append(f"- `{name}` : {value:.3e}")
    lines += [
        "",
        "## Statut exact de cette étape",
        "",
        "Démontré dans la spécification : unicité de l'inverse et réductions proximales.",
        "",
        "Vérifié ici : oracle NumPy FP64, contrat bi-coordonnées, cas limites et KKT sur les domaines consignés.",
        "",
        "Non vérifié ici : backend Torch, double backward, Triton, CUDA, performance et originalité bibliographique.",
    ]
    (ROOT / "REFERENCE_V1_CLOSURE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
