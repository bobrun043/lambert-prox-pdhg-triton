# Triton canonical patch V1.1 — Colab compiler hotfix

Date: 2026-07-30

## Failure reproduced from the CUDA run

Triton rejected expressions such as:

```python
if MODE == MODE_GAUSSIAN:
```

inside an `@triton.jit` function. `MODE_GAUSSIAN` was an ordinary Python
module global. Recent Triton compilers do not permit access to such globals
from JIT bodies unless they were instantiated as Triton constexpr objects.
Enabling `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` would only be a temporary,
unsupported workaround and is not used.

## Canonical correction

The mode contract remains:

| Literal | Mode |
|---:|---|
| 0 | Gaussian |
| 1 | Poisson intensity |
| 2 | Poisson log-intensity |
| 3 | KL |
| 4 | xlogx |
| 5 | exp |
| 6 | neglog |

Inside JIT kernels, all mode branches now compare the constexpr parameter
`MODE` directly with literals `0` through `5`; the final branch is mode `6`.
The symbolic constants remain only in ordinary Python wrappers and routing
maps, where they are safe and improve readability.

Patched files:

- `triton_prox_canonical_v1.py`
- `triton_stencil_canonical_v1.py`

## Regression guard

`check_triton_jit_globals_v1.py` parses every `@triton.jit` function and fails
if a `MODE_*` Python global is loaded from a JIT body.

Local checks executed in the CPU-only build environment:

- Python syntax/bytecode compilation: PASS
- JIT global static guard: PASS — 7 JIT functions checked
- CUDA compilation/runtime: NOT_RUN in this environment

The CUDA oracle validation must now be rerun on the A100. No promotion status
is inferred from these static checks.
