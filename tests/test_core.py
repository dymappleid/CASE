from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from case_stopping import apply_policy, load_policy
from case_stopping.core import OperatingPoint, RidgeModel, feature_vector
from case_stopping.fit import select_operating_point

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_feature_vector_has_released_dimension(self):
        row = json.loads((ROOT / "examples/example_trajectory.json").read_text())
        values = feature_vector(row, 0)
        self.assertEqual(values.shape, (18,))
        self.assertTrue(np.isfinite(values).all())

    def test_all_frozen_models_have_18_coefficients(self):
        payload = json.loads((ROOT / "configs/frozen_policies.json").read_text())
        self.assertEqual(sum(len(group) for group in payload["policies"].values()), 6)
        for group in payload["policies"].values():
            for item in group.values():
                model = RidgeModel.from_dict(item["ridge_model"])
                self.assertEqual(model.coefficients.shape, (18,))
                OperatingPoint(item["cost_weight"], item["boundary"], item["activation"])

    def test_public_policy_api(self):
        row = json.loads((ROOT / "examples/example_trajectory.json").read_text())
        policy = load_policy(
            ROOT / "configs/frozen_policies.json", "videoseek", "qwen35"
        )
        decision = apply_policy([row], policy)[0]
        self.assertEqual(decision["sample_id"], row["sample_id"])
        self.assertTrue(decision["activated"])

    def test_selector_strict_and_fallback_branches(self):
        strict_records = [
            {"model_token_saving": 0.56, "accuracy_lcb": -0.005, "accuracy_delta": 0.01},
            {"model_token_saving": 0.63, "accuracy_lcb": -0.008, "accuracy_delta": 0.00},
        ]
        _, selected, audit = select_operating_point(strict_records)
        self.assertEqual(audit["branch"], "strict")
        self.assertEqual(selected["model_token_saving"], 0.63)

        fallback_records = [
            {"model_token_saving": 0.39, "accuracy_lcb": -0.02, "accuracy_delta": 0.01},
            {"model_token_saving": 0.46, "accuracy_lcb": -0.02, "accuracy_delta": 0.02},
        ]
        _, selected, audit = select_operating_point(fallback_records)
        self.assertEqual(audit["branch"], "fallback")
        self.assertEqual(selected["model_token_saving"], 0.39)


if __name__ == "__main__":
    unittest.main()
