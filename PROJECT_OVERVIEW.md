# Project overview

This repository contains the `CLOSED_T4_ATTESTED` state of a correctness-first implementation of Wright-Omega/Lambert-type proximal operators and a periodic two-dimensional TV-PDHG solver with Torch and Triton backends.

## Established scope

- Stable value/log evaluation of the scalar equation `u + log(u) = R`.
- Seven canonical proximal models with explicit closed domains.
- Independent NumPy FP64 and Torch implementations.
- Exact finite-grid norm for the periodic forward gradient.
- End-of-run primal/dual objectives, Fenchel gap, dual feasibility, and fixed-point KKT residuals.
- Forward FP32 Triton kernels checked against Torch.
- Fused periodic-TV stencil kernels for complete PDHG updates.
- Conservative runtime routing with explicit Torch fallback.
- A source-bound Tesla T4 validation attestation and raw benchmark evidence.

## Status separation

- `stabilized`: the relative iterate-change stopping condition was met.
- `certified`: numerical gap/KKT/feasibility thresholds were met.
- mathematical convergence: a theorem-level statement for exact PDHG under its hypotheses; it is not inferred automatically from a finite-precision run with approximate prox evaluation.

## Validated accelerated environment

- Tesla T4, compute capability 7.5
- PyTorch `2.11.0+cu128`
- CUDA runtime `12.8`
- Triton `3.6.0`
- FP32 forward execution
- recorded models and geometries only

## Non-claims

The repository does not claim new proximal formulas, priority for Wright Omega/Lambert W, state-of-the-art speed, cross-GPU portability, Triton autograd, mixed-precision support, or a general finite-precision convergence theorem.

## Methodological context

The project was developed by an independent autodidact using AI systems for derivation, implementation, debugging, adversarial review, test design, and documentation. The scientific status rests on the exposed formulas and reproducible evidence, including failures and limits—not on AI authority.

## Current production profile

The public API is `pdhg_auto_routed_close1.pdhg_auto`. Torch remains normative outside the exact accelerated routing contract. Auto-Routed V3 is retained as a historical, hash-bound layer beneath CLOSE1.
