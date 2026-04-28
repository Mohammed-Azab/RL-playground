"""Repo-local interpreter startup tweaks.

Suppress the known TensorBoard pkg_resources deprecation warning without
changing runtime behavior.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\.",
    category=UserWarning,
)