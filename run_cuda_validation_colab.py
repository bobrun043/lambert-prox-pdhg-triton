"""No-argument CUDA runner suitable for a Colab cell or direct execution."""
from pathlib import Path
import json

from triton_oracle_validation_v2 import run
from triton_promotion_gate_v1 import load_and_validate

HERE = Path(__file__).resolve().parent
REPORT = HERE / "triton_validation_metrics_v2.json"
result = run(REPORT)
print(json.dumps(result, indent=2))
if result.get("status") != "PASS":
    raise SystemExit(f"CUDA validation did not pass: {result.get('status')}")
load_and_validate(REPORT)
print("PROMOTION_GATE: PASS")
