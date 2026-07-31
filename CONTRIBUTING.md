# Contributing

Contributions must preserve the repository's hierarchy of trust:

1. mathematical statement, closed domain, and explicit hypotheses;
2. independent FP64 reference;
3. Torch oracle and derivative checks;
4. exact finite-grid operator constants;
5. primal/dual objectives, gap, feasibility, and KKT diagnostics;
6. Triton equivalence and multi-iteration regression;
7. performance measurement separated from correctness;
8. environment-local routing only after source-bound statistical evidence.

Do not widen a claim, dtype, device, boundary condition, geometry, or model scope without adding explicit validation. Keep `stabilized`, numerical `certified`, and mathematical convergence as separate statuses. Explicit Triton requests must never silently fall back.
