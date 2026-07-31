from __future__ import annotations
import json
from pathlib import Path


def audit() -> dict:
    root = Path(__file__).resolve().parent
    elem = (root / "triton_prox_fused_candidate_v2.py").read_text(encoding="utf-8")
    stencil = (root / "triton_stencil_candidate_v5.py").read_text(encoding="utf-8")
    findings = [
        {
            "id": "T-AUDIT-001",
            "severity": "material",
            "scope": "elementwise+stencil",
            "finding": "Candidate Halley uses four value-coordinate iterations rather than the canonical six-pass bi-coordinate primitive.",
            "evidence": "for _ in range(4)" in elem and "iters: tl.constexpr = 4" in stencil,
            "consequence": "Equivalence must be established empirically on CUDA; it is not true by construction.",
        },
        {
            "id": "T-AUDIT-002",
            "severity": "material",
            "scope": "exp and poisson_log",
            "finding": "Candidate reconstructs v-u or d-u instead of canonical q-log(lambda).",
            "evidence": "t = v - w" in elem and "x_new = z - u" in stencil,
            "consequence": "Possible cancellation for large positive matched terms; wide-domain tests are mandatory.",
        },
        {
            "id": "T-AUDIT-003",
            "severity": "material",
            "scope": "KL",
            "finding": "Candidate replaces y=0 by an epsilon floor instead of implementing the exact zero boundary.",
            "evidence": "yobs = tl.maximum(yobs, eps)" in elem and "ysafe = tl.maximum(yobs_ij, 1e-30)" in stencil,
            "consequence": "The candidate computes a regularized problem at y=0, not the canonical prox.",
        },
        {
            "id": "T-AUDIT-004",
            "severity": "scope",
            "scope": "stencil",
            "finding": "Fused stencil candidate has no poisson_log mode.",
            "evidence": '"poisson_log"' not in stencil,
            "consequence": "Stencil coverage is six fidelity modes, not seven.",
        },
        {
            "id": "T-AUDIT-005",
            "severity": "positive",
            "scope": "stencil",
            "finding": "Candidate wrappers pass runtime strides and use distinct output buffers.",
            "evidence": "x_in.stride(0)" in stencil and "x_in.data_ptr() == x_out.data_ptr()" in stencil,
            "consequence": "The intended stride-safe and no-in-place contracts are visible in source, but still require CUDA execution.",
        },
    ]
    report = {"schema":"triton-static-audit-v1", "findings":findings,
              "all_evidence_patterns_found": all(f["evidence"] for f in findings)}
    (root / "triton_static_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    audit()
