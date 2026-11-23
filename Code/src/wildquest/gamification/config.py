"""Configurable XP and encounter rules."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/gamification.json"


@dataclass(frozen=True)
class GamificationConfig:
    # Below this classifier confidence an observation is stored as NEEDS_REVIEW (no XP).
    # Provisional: issue #14 will recommend a validation-based value.
    review_threshold: float = 0.70
    first_discovery_xp: int = 100
    encounter_xp: int = 10
    # Same user + same species within this time AND distance = one encounter (one reward).
    encounter_window_hours: float = 6.0
    encounter_radius_m: float = 500.0
    # Caps NEW_ENCOUNTER awards per user and species per UTC day of the server clock.
    max_encounter_awards_per_species_per_day: int = 3
    # Reject observations stamped further than this in the future (clock skew allowance).
    max_future_skew_minutes: float = 10.0

    def __post_init__(self):
        def check(name, ok):
            if not ok:
                raise ValueError(f"Invalid gamification setting {name}={getattr(self, name)!r}")

        for name in ("review_threshold", "encounter_window_hours", "encounter_radius_m",
                     "max_future_skew_minutes"):
            value = getattr(self, name)
            check(name, isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and value >= 0)
        check("review_threshold", self.review_threshold <= 1)
        for name in ("first_discovery_xp", "encounter_xp",
                     "max_encounter_awards_per_species_per_day"):
            value = getattr(self, name)
            check(name, isinstance(value, int) and not isinstance(value, bool) and value >= 0)

    @classmethod
    def from_dict(cls, data: dict) -> "GamificationConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown gamification setting(s): {', '.join(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG_PATH) -> "GamificationConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
