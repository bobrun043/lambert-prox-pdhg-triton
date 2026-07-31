"""Verify the autonomous CLOSE1 package without importing Torch or Triton."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "close1_package_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_hash_map(base: Path, files: dict[str, str]) -> list[str]:
    errors = []
    for name, expected in files.items():
        path = base / name
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif sha256_file(path) != expected:
            errors.append(f"hash mismatch: {name}")
    return errors


def release_digest(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print("FAIL: close1_package_manifest.json is missing")
        return 1
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema") != "lambert-prox-close1-package-v2":
        print("FAIL: unexpected CLOSE1 manifest schema")
        return 1
    errors = verify_hash_map(HERE, data.get("files", {}))

    t4_path = HERE / "triton_validation_metrics_close1.json"
    if not t4_path.is_file():
        errors.append("missing CLOSE1 T4 attestation")
    else:
        t4 = json.loads(t4_path.read_text(encoding="utf-8"))
        provenance = t4.get("provenance", {})
        before = provenance.get("source_hashes_before", {})
        after = provenance.get("source_hashes_after", {})
        summary = t4.get("validation", {}).get("summary", {})
        if t4.get("status") != "PASS":
            errors.append("CLOSE1 T4 attestation is not PASS")
        if before != after or not provenance.get("sources_unchanged_during_run"):
            errors.append("CLOSE1 T4 source maps differ")
        errors.extend(f"T4 source map: {error}" for error in verify_hash_map(HERE, before))
        if release_digest(before) != provenance.get("release_sha256"):
            errors.append("CLOSE1 T4 release digest mismatch")
        if summary.get("count") != 47 or summary.get("passed") != 47:
            errors.append("CLOSE1 T4 summary is not 47/47")
        if t4.get("validation", {}).get("environment", {}).get("device") != "Tesla T4":
            errors.append("CLOSE1 T4 device is not Tesla T4")

    historical = {
        "promotion_manifest_v1.json": "sources",
        "routing_evidence_manifest_v1.json": "files",
    }
    for manifest_name, field in historical.items():
        path = HERE / manifest_name
        if not path.is_file():
            errors.append(f"missing historical manifest: {manifest_name}")
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(
            f"{manifest_name}: {error}"
            for error in verify_hash_map(HERE, item.get(field, {}))
        )

    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    print(json.dumps({
        "status": "PASS",
        "close1_files_verified": len(data["files"]),
        "historical_manifests_checked": len(historical),
        "t4_close1_attestation": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
