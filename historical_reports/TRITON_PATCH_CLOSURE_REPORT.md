# Lambert-Prox — Triton Canonical Patch V1

Date: 2026-07-30  
Status: **STATIC_AND_CPU_PASS_CUDA_NOT_RUN**

## 1. Purpose

This stage repairs the four deviations previously identified between the
legacy Triton candidates and the unique mathematical/Torch specification. It
does not claim that the kernels have compiled or passed on CUDA in the current
environment.

## 2. Corrections implemented

### C1 — Canonical primitive

Both elementwise and fused-stencil candidates now implement the fixed six-pass
bi-coordinate primitive

\[
u+\log u=R,\qquad q=\log u,
\]

with a value-coordinate branch for ordinary inputs and a log-coordinate branch
for `R < -8`.

### C2 — Cancellation-safe reconstruction

The `exp` and Poisson log-intensity proxes are reconstructed by

\[
x=q-\log\lambda,
\]

rather than by subtracting two potentially large matched quantities such as
`v-u` or `d-u`.

### C3 — Exact KL boundary

The KL candidates implement the mathematical boundary

\[
y=0 \Longrightarrow \operatorname{prox}(v;y,\lambda)=0,
\]

instead of replacing zero by an epsilon and solving a different regularized
problem.

### C4 — Seventh fused mode

The primal fused stencil now contains a distinct `poisson_log` mode. Stencil
coverage is therefore seven fidelity proxes at source level:

- Gaussian;
- Poisson intensity;
- Poisson log-intensity;
- KL;
- `xlogx`;
- `exp`;
- `neglog`.

## 3. Additional conservative decisions

The corrected candidates are:

- CUDA `float32` only;
- forward-only;
- explicit about unavailable CUDA/Triton;
- free of silent fallbacks;
- stride-aware;
- protected against in-place primal updates.

No backward/autograd implementation was retained from the legacy candidate,
because it has not been audited against the canonical derivative contracts.
No AMP, FP16, BF16, FP64-Triton, speedup, SOTA or production claim is made.

## 4. Executed validation

### Static source audit

Result: **PASS**.

The audit confirms all four corrections, the absence of the forbidden `v-u`
and `d-u` reconstructions, the no-in-place contract and the forward-only scope.

### CPU formula mirror

Five wide-domain tests compare a CPU mirror of the exact formulas encoded in
the candidates with the unique Torch oracle.

Result: **5/5 PASS**.

The tested range includes `R in [-100,100]`, exact KL zeros, and the
cancellation-sensitive `exp` and Poisson-log reconstructions.

### Canonical Torch/PDHG regression

Result: **33/33 tests PASS** in the complete pack.

The existing five well-posed PDHG models remain finite, deterministic and
objective-decreasing. The FP64 adjoint relative gap remains

\[
2.4023500583132644\times10^{-15}.
\]

### CUDA/Triton

Current result:

```text
status: NOT_RUN
reason: CUDA and Triton are both required
```

Environment:

- PyTorch `2.10.0+cpu`;
- CUDA unavailable;
- Triton not installed.

The strict `--require-cuda` command exits with code **3**, proving that an
absent CUDA run cannot be confused with a pass.

## 5. CUDA acceptance battery

The prepared battery contains **47 required comparisons**:

- seven standard elementwise proxes;
- exact KL zero-boundary test;
- two wide cancellation-sensitive tests;
- seven non-contiguous elementwise tests;
- dual fused stencil;
- seven primal fused modes;
- non-contiguous dual fused stencil;
- seven non-contiguous primal fused modes;
- complete dual and primal PDHG-step comparisons for all seven proxes.

Acceptance requires every comparison to satisfy

\[
\max_i\frac{|x_i^{\rm Triton}-x_i^{\rm Torch}|}
{\max(1,|x_i^{\rm Triton}|,|x_i^{\rm Torch}|)}\le 8\times10^{-5}.
\]

## 6. Promotion gate

`triton_promotion_gate_v1.py` refuses integration unless the report:

- uses schema `triton-oracle-validation-v2`;
- has status `PASS`;
- contains at least 47 tests;
- contains no failed comparison;
- records CUDA and Triton as available.

The current `NOT_RUN` report is therefore correctly rejected. The corrected
kernels are **not yet promoted** as the PDHG default.

## 7. Exact remaining step

On a CUDA environment with a matching Triton installation, run:

```bash
python run_cuda_validation_colab.py
```

Only a final `PROMOTION_GATE: PASS` permits reconnection of the corrected
modules to the accelerated PDHG backend. Performance benchmarking comes after
this correctness gate, not before it.

---

## V1.1 Colab compiler hotfix — 2026-07-30

The first CUDA compilation exposed an incompatibility with recent Triton:
ordinary Python globals such as `MODE_GAUSSIAN` cannot be referenced inside
`@triton.jit` bodies. The JIT branches now use constexpr integer literals
`0..6`. Symbolic mode constants remain host-side only. See
`TRITON_PATCH_V1_1_HOTFIX_REPORT.md`.
