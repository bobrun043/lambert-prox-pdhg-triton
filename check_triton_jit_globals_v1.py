#!/usr/bin/env python3
"""Static guard for unsupported Python globals inside Triton @jit functions.

This does not replace CUDA compilation. It blocks the concrete regression found
on Colab/Triton: symbolic MODE_* globals referenced from a @triton.jit body.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

FILES = [
    Path(__file__).with_name("triton_prox_canonical_v1.py"),
    Path(__file__).with_name("triton_stencil_canonical_v1.py"),
]
FORBIDDEN_PREFIXES = ("MODE_",)


def is_triton_jit(dec: ast.expr) -> bool:
    return (
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "triton"
        and dec.attr == "jit"
    )


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(is_triton_jit(d) for d in node.decorator_list):
                continue
            checked += 1
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id.startswith(FORBIDDEN_PREFIXES):
                        failures.append(
                            f"{path.name}:{sub.lineno}: {node.name} loads unsupported global {sub.id}"
                        )
    if failures:
        print("TRITON_JIT_GLOBAL_GUARD: FAIL")
        print("\n".join(failures))
        return 1
    print(f"TRITON_JIT_GLOBAL_GUARD: PASS ({checked} @triton.jit functions checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
