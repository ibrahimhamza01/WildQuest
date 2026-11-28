"""XP rules, encounter deduplication, and the observation service (issue #12)."""

from .config import DEFAULT_CONFIG_PATH, GamificationConfig
from .rules import haversine_m, is_same_encounter, matching_encounter
from .service import ObservationService

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "GamificationConfig",
    "ObservationService",
    "haversine_m",
    "is_same_encounter",
    "matching_encounter",
]
