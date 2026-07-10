"""``ml_xray.embed`` -- post-training embedding diff.

Phase 3 (scaffolded). Public API: :class:`EmbeddingDiff`, :class:`EmbedDiffReport`.
"""

from __future__ import annotations

from .diff import EmbedDiffReport, EmbeddingDiff

__all__ = ["EmbeddingDiff", "EmbedDiffReport"]
