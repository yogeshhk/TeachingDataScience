"""Configuration for stlinspector's validation thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Tunable thresholds used by the inspection engine."""

    degenerate_face_area_threshold: float = 1e-8


DEFAULT_CONFIG = Config()
