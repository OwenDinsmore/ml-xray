"""``ml_xray.slices`` -- post-training error analysis / slice discovery.

Phase 2 (scaffolded). Public API: :class:`SliceFinder`, :class:`SliceReport`,
:class:`Slice`.
"""

from __future__ import annotations

from .finder import Slice, SliceFinder, SliceReport

__all__ = ["SliceFinder", "SliceReport", "Slice"]
