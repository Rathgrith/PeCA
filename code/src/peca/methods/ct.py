"""Cyclic-gated Temporal Fusion (CT), Section 3 of the PeCA paper."""

from __future__ import annotations

from peca.core import _ttc_calibrate_clip, _ttc_step


def fuse_adjacent_probabilities(
    current_probabilities,
    current_features,
    adjacent_probabilities,
    adjacent_features,
    current_palette,
    adjacent_palette,
    gamma=1.0,
    eps=1.0e-8,
    cycle_consistency=True,
    palette_tolerance=1.0e-5,
):
    """Fuse one frame with a cycle-consistent adjacent-frame prior."""

    return _ttc_step(
        current_probabilities,
        current_features,
        adjacent_probabilities,
        adjacent_features,
        current_palette,
        adjacent_palette,
        gamma=gamma,
        eps=eps,
        use_cycle_consistency=cycle_consistency,
        palette_tol=palette_tolerance,
    )


def fuse_sequence_probabilities(
    probabilities,
    features,
    palettes,
    gamma=1.0,
    num_sweeps=1,
    bidirectional=True,
    cycle_consistency=True,
    eps=1.0e-8,
    palette_tolerance=1.0e-5,
):
    """Apply the paper's one-sweep bidirectional CT update to a clip."""

    return _ttc_calibrate_clip(
        probabilities,
        features,
        palettes,
        gamma=gamma,
        num_sweeps=num_sweeps,
        bidirectional=bidirectional,
        use_cycle_consistency=cycle_consistency,
        eps=eps,
        palette_tol=palette_tolerance,
    )


__all__ = ["fuse_adjacent_probabilities", "fuse_sequence_probabilities"]
