"""Rebuild the deterministic CLOSE1 file manifest and SHA256SUMS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "close1_package_manifest.json"
SUMS = HERE / "SHA256SUMS.txt"
EXCLUDED_NAMES = {
    MANIFEST.name,
    SUMS.name,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def included_files() -> list[Path]:
    return sorted(
        path for path in HERE.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    )


def main() -> int:
    files = {
        path.relative_to(HERE).as_posix(): sha256_file(path)
        for path in included_files()
    }
    manifest = {
        "schema": "lambert-prox-close1-package-v2",
        "date": "2026-07-31",
        "status": "CLOSED_T4_ATTESTED",
        "scope": (
            "Autonomous source, tests, historical evidence, CLOSE1 diagnostics, "
            "and a source-bound Tesla T4 validation attestation."
        ),
        "t4_close1_execution": "PASS_TESLA_T4",
        "t4_release_sha256": "7e0757258621520664c48653650efdf5e13b6ed95773537dce40133172d2417f",
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    SUMS.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in files.items()),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
