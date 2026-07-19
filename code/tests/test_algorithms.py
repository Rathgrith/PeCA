from __future__ import annotations

import unittest

import torch

from peca.methods import (
    ReferenceViewAugmenter,
    aggregate_palette_probabilities,
    fuse_adjacent_probabilities,
)
from peca.utils.visualization import tensor_to_uint8_image


class AlgorithmTests(unittest.TestCase):
    def test_pa_aggregates_repeated_palette_entries(self):
        similarities = torch.tensor([[4.0, 4.0, 4.3]])
        red = torch.tensor([1.0, 0.0, 0.0, 1.0])
        blue = torch.tensor([0.0, 0.0, 1.0, 1.0])
        source_colours = torch.stack([red, red, blue])

        _, prediction, probabilities = aggregate_palette_probabilities(
            similarities,
            source_colours,
            temperature=1.0,
            top_k=3,
        )
        torch.testing.assert_close(prediction[0], red)
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))

        _, top_one_prediction, _ = aggregate_palette_probabilities(
            similarities,
            source_colours,
            temperature=1.0,
            top_k=1,
        )
        torch.testing.assert_close(top_one_prediction[0], blue)

    def test_ct_multiplies_cycle_consistent_adjacent_prior(self):
        current = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        adjacent = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
        features = torch.eye(2)
        palette = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        calibrated = fuse_adjacent_probabilities(
            current,
            features,
            adjacent,
            features,
            palette,
            palette,
            gamma=1.0,
        )
        expected = torch.tensor([[0.08 / 0.26, 0.18 / 0.26], [0.14 / 0.38, 0.24 / 0.38]])
        torch.testing.assert_close(calibrated, expected)

    def test_are_identity_transform_preserves_aligned_inputs(self):
        generator = ReferenceViewAugmenter(
            flip_p=0.0,
            vflip_p=0.0,
            rotate90_p=0.0,
            affine_p=0.0,
        )
        line = torch.rand(3, 8, 8)
        segments = torch.arange(64).reshape(8, 8)
        colour = torch.rand(3, 8, 8)
        line_out, segments_out, colour_out = generator(line, segments, colour)
        torch.testing.assert_close(line_out, line)
        torch.testing.assert_close(segments_out, segments)
        torch.testing.assert_close(colour_out, colour)

    def test_visualization_helper_returns_rgb_uint8(self):
        image = tensor_to_uint8_image(torch.ones(4, 3, 2))
        self.assertEqual(image.shape, (3, 2, 3))
        self.assertEqual(image.dtype.name, "uint8")


if __name__ == "__main__":
    unittest.main()
