from __future__ import annotations

import unittest
from pathlib import Path

from peca.config import PAPER_DEFAULTS, checkpoint_path, load_config, to_runtime_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_matches_main_paper(self):
        config = load_config(ROOT / "configs/default.yaml")
        runtime = to_runtime_config(config)

        self.assertEqual(config["method"]["name"], "peca")
        self.assertEqual(config["backbone"]["name"], "sam2_1_large")
        self.assertTrue(config["backbone"]["training_free"])
        self.assertIsNone(checkpoint_path(config))
        self.assertEqual(runtime["memory"]["feature_source"], "sam2")
        self.assertEqual(runtime["memory"]["sam2"]["input_size"], [512, 512])

        self.assertEqual(config["method"]["pa"]["top_k"], PAPER_DEFAULTS["pa_top_k"])
        self.assertEqual(
            config["method"]["pa"]["temperature"],
            PAPER_DEFAULTS["pa_temperature"],
        )
        self.assertEqual(config["method"]["are"]["num_views"], PAPER_DEFAULTS["are_num_views"])
        self.assertEqual(
            config["method"]["are"]["candidate_multiplier"],
            PAPER_DEFAULTS["are_candidate_multiplier"],
        )
        self.assertEqual(config["method"]["ct"]["gamma"], PAPER_DEFAULTS["ct_gamma"])

        self.assertEqual(runtime["inference"]["infer_topk_src"], 64)
        self.assertEqual(runtime["inference"]["infer_tau"], 0.05)
        self.assertEqual(runtime["memory"]["ref_aug"]["num_views"], 31)
        self.assertEqual(runtime["memory"]["active_memory"]["candidate_multiplier"], 4)
        self.assertIsNone(config["experiment"]["seed"])
        self.assertIsNone(runtime["memory"]["active_memory"]["seed"])
        self.assertEqual(runtime["inference"]["temporal_calibration"]["gamma"], 1.0)

    def test_base_disables_all_three_components(self):
        config = load_config(
            ROOT / "configs/default.yaml",
            components={"backbone": "dinov2_vitl14", "method": "base"},
        )
        runtime = to_runtime_config(config)
        self.assertEqual(
            runtime["network"]["dino_repository"],
            "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        )
        self.assertFalse(runtime["memory"]["active_memory"]["enable"])
        self.assertFalse(runtime["memory"]["ref_aug"]["enable"])
        self.assertFalse(runtime["inference"]["infer_use_color_agg"])
        self.assertFalse(runtime["inference"]["temporal_calibration"]["enable"])

    def test_dacon_is_the_only_release_checkpoint(self):
        dacon = load_config(
            ROOT / "configs/default.yaml",
            components={"backbone": "dacon_v1_1"},
        )
        self.assertFalse(dacon["backbone"]["training_free"])
        self.assertEqual(checkpoint_path(dacon), "checkpoints/dacon_v1_1.pth")

    def test_component_switches_replace_complete_sections(self):
        config = load_config(
            ROOT / "configs/default.yaml",
            components={
                "backbone": "dinov2_vitl14",
                "dataset": "anita_pirate",
                "protocol": "inbetween_anita_pirate",
            },
        )
        runtime = to_runtime_config(config)
        self.assertEqual(config["dataset"]["name"], "anita_pirate")
        self.assertEqual(config["backbone"]["name"], "dinov2_vitl14")
        self.assertEqual(
            config["experiment"]["name"],
            "anita_pirate_inbetween_dinov2_vitl14_peca",
        )
        self.assertEqual(runtime["datasets"]["clip_interval"], "max")
        self.assertEqual(runtime["memory"]["feature_source"], "dino")

    def test_reference_count_protocols(self):
        five_shot = load_config(
            ROOT / "configs/default.yaml",
            components={"protocol": "design_sheet_5shot"},
        )
        max_shot = load_config(
            ROOT / "configs/default.yaml",
            components={"protocol": "design_sheet_maxshot"},
        )
        self.assertEqual(five_shot["protocol"]["ref_shots"], 5)
        self.assertEqual(max_shot["protocol"]["ref_shots"], "max")


if __name__ == "__main__":
    unittest.main()
