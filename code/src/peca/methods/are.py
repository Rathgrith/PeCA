"""Active Reference Expansion (ARE), Section 3 of the PeCA paper."""

from __future__ import annotations

from peca.core import RefAugmenter, _select_active_ref_views


class ReferenceViewAugmenter(RefAugmenter):
    """Generate line/segment/colour-consistent reference transformations.

    ``configs/methods/peca.yaml`` contains the exact transform distribution
    used for the main-paper experiments. The original reference is always kept
    in addition to the ``B`` selected augmented views.
    """


def select_active_references(*args, **kwargs):
    """Select ARE views greedily by target-feature coverage.

    The runner uses this small paper-facing wrapper while the implementation is
    shared with the compatibility core used for the reported experiments.
    """

    return _select_active_ref_views(*args, **kwargs)


__all__ = ["ReferenceViewAugmenter", "select_active_references"]
