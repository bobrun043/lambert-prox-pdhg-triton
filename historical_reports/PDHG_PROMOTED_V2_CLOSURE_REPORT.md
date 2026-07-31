# Closure report — PDHG Promoted V2

Date: 2026-07-30  
Status: **PROMOTED_CORRECTNESS_ONLY**

## 1. Promotion decision

The CUDA promotion gate is passed. The supplied report records PyTorch
`2.11.0+cu128`, CUDA `12.8`, a Tesla T4, and an importable Triton runtime. All
47 required comparisons pass.

The numerical acceptance threshold was

\[
\max_i\frac{|x_i^{\rm Triton}-x_i^{\rm Torch}|}
{\max(1,|x_i^{\rm Triton}|,|x_i^{\rm Torch}|)}\le 8\times10^{-5}.
\]

The measured worst value is

\[
4.76837158203125\times 10^{-7},
\]

about 168 times below that threshold. The worst case is the wide-domain
Poisson log-intensity elementwise test; its maximum absolute discrepancy is
`7.62939453125e-06`.

## 2. Coverage of the CUDA evidence

The 47 successful comparisons comprise:

- 7 standard elementwise proxes;
- 1 exact KL zero-boundary test;
- 2 wide-domain cancellation tests;
- 7 non-contiguous elementwise tests;
- 1 fused dual stencil test;
- 7 fused primal stencil modes;
- 1 non-contiguous dual stencil test;
- 7 non-contiguous primal stencil modes;
- 7 complete PDHG dual-step comparisons;
- 7 complete PDHG primal-step comparisons.

This establishes forward FP32 numerical agreement with the Torch oracle for
the tested distributions and shapes. It does not establish a universal error
bound over all floating-point inputs.

## 3. Reconnection performed

`pdhg_promoted_v2.py` now routes explicitly to:

- `torch`;
- `triton_elementwise`;
- `triton_stencil`.

The promoted module names replace all legacy candidate names. Poisson
log-intensity uses the canonical `prox_poisson_log_axpy` entry point.

There is no automatic backend choice and no silent fallback. Triton execution
requires:

- a verified promotion manifest;
- unchanged SHA-256 hashes for the report and source files;
- CUDA and Triton availability;
- CUDA FP32 tensors.

## 4. Independent local regression

The CPU environment cannot execute Triton, but it validates the surrounding
integration:

- promotion manifest: PASS;
- source/report hashes: PASS;
- Python compilation: PASS;
- five well-posed Torch PDHG models: finite and objective-decreasing;
- invalid autonomous `exp+TV` and `-log+TV` models: rejected;
- unavailable Triton backend: explicit error;
- 20 assertion groups: PASS;
- FP64 adjoint relative gap:
  `2.4023500583132644e-15`.

## 5. Provenance status

A limitation remains visible. The original CUDA report does not contain hashes
of the source files that generated it. The new manifest is therefore a
**post-run package binding**: it prevents later unnoticed modification of the
promoted pack, but it is not cryptographic attestation that the original
Colab run used those exact bytes.

This is not grounds to reject the numerical result, given the continuous test
workflow and the reported gate pass, but it prevents calling the provenance
fully closed. The next validation format should include source hashes before
kernel compilation.

## 6. Claims allowed and forbidden

Allowed:

> On the reported Tesla T4 environment, the canonical forward FP32 Triton
> elementwise and fused-stencil kernels matched the unique Torch oracle in all
> 47 prescribed tests, with worst scaled discrepancy below `4.77e-7`.

Not allowed yet:

- fastest or SOTA implementation;
- speedup over Torch;
- production readiness;
- correctness of backward/autograd;
- FP16, BF16, AMP or Triton FP64 support;
- identical behavior on A100, RTX 3060 or other GPUs;
- universal accuracy outside the tested domains.

## 7. Next controlled stage

Run `run_promoted_regression_colab.py`. It compares complete multi-iteration
PDHG trajectories for the five well-posed models across Torch, elementwise
Triton and fused-stencil Triton. Timings are recorded descriptively but do not
become a performance claim until warmup, repetitions, synchronization,
confidence intervals and cross-device runs are added.
