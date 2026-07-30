# Lambert-Prox PDHG Triton

Correctness-first proximal operators of Lambert/entropy type, a canonical PyTorch PDHG solver, fused Triton kernels, and evidence-backed runtime routing for a validated Tesla T4 environment.

This repository is a research-software case study built by an independent autodidact with extensive AI assistance. The code and claims were progressively narrowed through independent references, KKT checks, adjoint tests, CUDA oracle comparisons, multi-iteration regression, statistical benchmarking, and SHA-256 evidence manifests.

## What is included

- stable solution of `u + log(u) = R` without forming `exp(R)`;
- canonical prox operators for Gaussian, exponential, `x log x`, KL, Poisson log-intensity, Poisson intensity, and `-log` models;
- NumPy FP64 reference implementation;
- PyTorch implementation with first- and second-order autograd validation;
- periodic 2D TV-PDHG reference solver;
- elementwise and fused stencil Triton forward kernels;
- conservative automatic routing with explicit Torch fallback;
- CUDA correctness reports, T4 benchmark evidence, source hashes, and a consolidated technical document.

## Validated scope

The accelerated route is validated for:

- GPU: Tesla T4, compute capability 7.5;
- PyTorch: 2.11.0+cu128;
- CUDA runtime: 12.8;
- Triton: 3.6.0;
- dtype: FP32;
- direction: forward only;
- models: Gaussian, Poisson intensity, Poisson log-intensity, KL, and `x log x`;
- tested sizes: `128x192` and `512x512`.

On that exact environment, 47/47 Torch–Triton oracle comparisons passed. Five multi-iteration PDHG trajectories passed. The paired bootstrap routing benchmark selected the fused stencil backend for all ten tested model-size pairs.

## Claims boundary

This repository does **not** claim:

- a new Lambert W function or a new proximal operator formula;
- mathematical priority for PDHG, TV regularization, or GPU fusion;
- state-of-the-art performance;
- cross-GPU performance portability;
- Triton autograd, AMP, FP16, BF16, or FP64 support;
- production readiness outside the validated scope.

The potentially original part is the bounded engineering architecture: a bi-coordinate log-domain primitive, a single canonical prox table, independent references, promoted Triton kernels, cryptographically bound evidence, and conservative device-local routing.

## Quick start

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python tests/test_pdhg_promoted_v2_cpu.py
PYTHONPATH=. python tests/test_pdhg_auto_routed_v3_cpu.py
PYTHONPATH=. python tests/test_routing_policy_v2.py
```

T4 smoke test:

```bash
python run_auto_routing_smoke_colab.py
```

Minimal usage:

```python
import torch
from pdhg_auto_routed_v3 import pdhg_auto

y = torch.rand((1, 1, 128, 192), device="cuda", dtype=torch.float32) + 0.1
x0 = y.clone()

x, dual, info, route = pdhg_auto(
    "gaussian", x0, y, lam_tv=0.08,
    max_iter=100, min_iter=100, tol=0.0,
)
print(route.executed_backend)
```

Explicit Triton requests never silently fall back. In `auto` mode, any mismatch in GPU, software versions, dtype, geometry, or evidence hashes returns to Torch and records the reason.

## Repository map

- `lambert_prox_reference_v1.py` — independent NumPy FP64 reference;
- `lambert_prox_torch_v1.py` — canonical Torch prox backend;
- `triton_prox_canonical_v1.py` — elementwise Triton kernels;
- `triton_stencil_canonical_v1.py` — fused periodic TV stencil kernels;
- `pdhg_promoted_v2.py` — explicit backend PDHG;
- `pdhg_auto_routed_v3.py` — conservative automatic router;
- `routing_policy_v2.py` — environment and evidence checks;
- `benchmark_promoted_routing_v1.py` — paired statistical benchmark;
- `tests/` — CPU policy and integration tests;
- `docs/` — closure reports;
- JSON/Markdown files at repository root — evidence required by the runtime manifests.

## Reproducibility and provenance

`promotion_manifest_v1.json` binds the promoted correctness package. `routing_evidence_manifest_v1.json` binds the T4 routing table, raw benchmark, summary, and relevant sources. The original CUDA correctness report did not embed source hashes during execution; this limitation is stated rather than hidden.

## AI assistance

AI systems were used for mathematical reformulation, code generation, debugging, adversarial review, test design, documentation, and consolidation. Numerical results were produced by the scripts and environments recorded in the evidence files. AI assistance is part of the development method, not evidence that every intermediate proposal was correct.

## Citation

See `CITATION.cff`.

## License

Apache License 2.0. See `LICENSE`.
