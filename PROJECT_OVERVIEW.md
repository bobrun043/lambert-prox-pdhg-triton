# Project overview

This repository documents a correctness-first implementation of Lambert-type proximal operators and a periodic 2D PDHG solver with PyTorch and Triton backends.

## What is established

- A canonical scalar primitive for solving `u + log(u) = R` with value and log coordinates.
- Canonical proximal operators for exponential, entropy, KL, negative-log and Poisson models.
- A PyTorch reference backend with native first- and second-order autograd on the tested scope.
- Forward FP32 Triton kernels validated against the canonical Torch backend.
- Fused periodic-TV stencil kernels for complete PDHG updates.
- A Tesla T4 routing table selected by paired bootstrap timing tests.
- Explicit fallback to Torch outside the tested device, dtype, model and image-size domain.

## Validated environment

- Tesla T4, compute capability 7.5
- PyTorch 2.11.0+cu128
- CUDA 12.8
- Triton 3.6.0
- FP32 forward execution

## Non-claims

This repository does not claim mathematical priority for Lambert-W proximal formulas, state-of-the-art performance, cross-GPU portability, validated AMP/half precision, Triton backward kernels, or a general-purpose imaging library.

## Methodological context

The project was developed by an independent autodidact using AI systems as tools for derivation, criticism, implementation, testing and repeated contradiction. The scientific status rests on the exposed formulas, tests, reports and reproducible artifacts—not on the origin of the author or on AI-generated authority.

## Current status

The current production profile is the auto-routed V3 package. Torch remains the normative fallback. Triton is selected only when the recorded routing contract is satisfied.
