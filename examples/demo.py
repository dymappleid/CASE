from __future__ import annotations

import json
from pathlib import Path

from case_stopping import apply_policy, load_policy

ROOT = Path(__file__).resolve().parents[1]
trajectory = json.loads((ROOT / "examples/example_trajectory.json").read_text())
policy = load_policy(
    ROOT / "configs/frozen_policies.json", "videoseek", "qwen35"
)

print(json.dumps(apply_policy([trajectory], policy)[0], indent=2))
