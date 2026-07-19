"""Probability Aggregation (PA), Section 3 of the PeCA paper."""

from __future__ import annotations

from peca.core import _color_aggregate_with_probs


def aggregate_palette_probabilities(similarities, source_colours, temperature=0.05, top_k=64):
    """Aggregate source-segment probabilities that share a palette colour.

    Returns representative source indices, predicted RGB colours, and the
    target-by-palette probability matrix.
    """

    return _color_aggregate_with_probs(
        similarities,
        source_colours,
        tau=temperature,
        topk_src=top_k,
    )


__all__ = ["aggregate_palette_probabilities"]
