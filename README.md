# Lambert-Prox PDHG Triton — CLOSE1

[![CPU validation](https://github.com/bobrun043/lambert-prox-pdhg-triton/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/bobrun043/lambert-prox-pdhg-triton/actions/workflows/cpu-tests.yml)

Correctness-first Lambert/Wright-\(\Omega\) proximal operators, a periodic 2-D TV-PDHG solver, fused Triton kernels, and conservative evidence-backed routing. `CLOSE1` is the source-bound release closed on 31 July 2026.

This is independent research software developed by Laurent Alexandre Hamza, an autodidact, with extensive AI assistance. Its status rests on exposed formulas, tests, raw metrics, source hashes, and explicit claim boundaries.

## What is established

| Layer | Recorded result |
|---|---|
| Scalar primitive | Stable bi-coordinate evaluation of `u + log(u) = R`, i.e. Wright \(\Omega(R)=W_0(e^R)\) |
| Prox catalogue | Gaussian, exponential, `x log x`, KL, Poisson log-intensity, Poisson intensity, and `-log` |
| CPU CLOSE1 | 58 tests passed; 5/5 controlled models stabilized and met numerical gap/KKT thresholds |
| Periodic gradient | Exact finite-grid norm; `sqrt(8)` retained only as a uniform upper bound |
| Tesla T4 | 47/47 Torch–Triton comparisons passed; worst scaled discrepancy `4.76837158203125e-07` |
| Provenance | Eight CUDA validation sources matched before and after the run |
| T4 routing study | 10/10 tested model-size pairs selected the fused stencil; minimum bootstrap CI lower bound `2.2247x` |

The CUDA attestation is limited to Tesla T4, compute capability 7.5, PyTorch `2.11.0+cu128`, CUDA `12.8`, Triton `3.6.0`, FP32 forward execution, and the recorded geometries and models. Torch is the normative fallback outside that contract.

## CLOSE1 corrections

The previous repository state exposed Auto-Routed V3. CLOSE1 keeps those historical artifacts byte-stable and adds:

- the exact periodic finite-grid norm;
- primal and dual objectives, Fenchel gap, dual feasibility, and normalized fixed-point KKT residuals;
- separate `stabilized` and `certified` flags;
- the closed domain `x >= 0` with `0 log 0 = 0` for `x log x`;
- a complete autonomous manifest;
- a Tesla T4 rerun with SHA-256 maps captured before import and checked again after execution.

`certified` is a numerical acceptance flag for the supplied tolerances. It is not a proof of convergence in floating-point arithmetic.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python verify_close1_package.py
python -m pytest -q
python run_close1_validation.py
```

Minimal use:

```python
import torch
from pdhg_auto_routed_close1 import pdhg_auto

y = torch.rand((1, 1, 128, 192), device="cuda", dtype=torch.float32) + 0.1
x0 = y.clone()

x, dual, info, route = pdhg_auto(
    "gaussian",
    x0,
    y,
    lam_tv=0.08,
    backend="auto",
    max_iter=200,
)

print(route.to_dict())
print(info.diagnostics.to_dict())
```

An explicit Triton request never silently falls back. In `auto` mode, an environment, evidence, model, dtype, or geometry mismatch selects Torch and records the reason.

## Repository map

- `lambert_prox_reference_v1.py` — independent NumPy/FP64 reference;
- `lambert_prox_torch_v1.py` — canonical Torch proximal backend;
- `pdhg_close1.py` — exact grid norm and CLOSE1 gap/KKT diagnostics;
- `pdhg_auto_routed_close1.py` — public evidence-backed API;
- `triton_prox_canonical_v1.py` and `triton_stencil_canonical_v1.py` — promoted Triton kernels;
- `triton_validation_metrics_close1.json` — cleaned T4 attestation;
- `t4_close1_colab_stdout_2026-07-31.txt` — raw Colab output;
- `close1_package_manifest.json` and `SHA256SUMS.txt` — repository integrity map;
- `historical_reports/` — earlier reports, explicitly separated from CLOSE1;
- `docs/` — the consolidated 18-page formalisation.

## Reproducibility

The normative local commands are:

```bash
python verify_close1_package.py
python -m pytest -q
python run_close1_validation.py
```

The source-bound T4 rerun is:

```python
%run run_cuda_validation_close1_colab.py
```

Its release digest is:

```text
7e0757258621520664c48653650efdf5e13b6ed95773537dce40133172d2417f
```

## Claims boundary

This repository does **not** claim a new special function, new proximal formulas, mathematical priority for PDHG or TV regularization, state-of-the-art speed, inter-GPU portability, backward Triton kernels, AMP/FP16/BF16/FP64 validation, or a general proof for inexact finite-precision PDHG.

The contribution is the bounded integration: one bi-coordinate primitive, one canonical prox contract, independent references, exact finite-grid diagnostics, promoted GPU kernels, source-bound evidence, and conservative device-local routing.

See [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), [`CLOSE1_CLOSURE_REPORT.md`](CLOSE1_CLOSURE_REPORT.md), and the [consolidated formalisation](docs/LAMBERT_PROX_CLOSE1_FORMALISATION_CONSOLIDEE_2026-07-31.pdf).

## License and citation

Apache License 2.0. See `LICENSE`, `NOTICE.md`, and `CITATION.cff`.
