from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_videomme_split_is_exact_and_disjoint(self):
        split = json.loads((ROOT / "configs/videomme_split.json").read_text())
        train = set(split["train_video_ids"])
        calibration = set(split["calibration_video_ids"])
        evaluation = set(split["evaluation_video_ids"])
        self.assertEqual((len(train), len(calibration), len(evaluation)), (50, 100, 750))
        self.assertFalse(train & calibration or train & evaluation or calibration & evaluation)
        self.assertEqual(len(train | calibration | evaluation), 900)

    def test_released_operating_points(self):
        policies = json.loads((ROOT / "configs/frozen_policies.json").read_text())["policies"]
        expected = {
            ("videoseek", "qwen35"): (0.2, 0.195),
            ("videoseek", "glm46v"): (0.2, -0.01),
            ("videoseek", "internvl35"): (0.8, -0.175),
            ("avp", "qwen35"): (0.4, 0.07),
            ("avp", "glm46v"): (0.05, 0.07),
            ("avp", "internvl35"): (0.8, -0.19),
        }
        for (host, model), (weight, boundary) in expected.items():
            item = policies[host][model]
            self.assertAlmostEqual(item["cost_weight"], weight)
            self.assertAlmostEqual(item["boundary"], boundary)
            self.assertEqual(item["activation"], "medium_long")




if __name__ == "__main__":
    unittest.main()
