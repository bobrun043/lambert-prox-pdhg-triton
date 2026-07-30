# Contributing

Contributions should preserve the repository's hierarchy of trust:

1. mathematical statement and domain;
2. independent FP64 reference;
3. Torch oracle;
4. Triton equivalence;
5. multi-iteration PDHG regression;
6. performance measurement separated from correctness;
7. environment-local routing only after robust statistical evidence.

Do not widen a claim, dtype, device, boundary condition, or model scope without adding explicit validation. Explicit Triton requests must never silently fall back.
