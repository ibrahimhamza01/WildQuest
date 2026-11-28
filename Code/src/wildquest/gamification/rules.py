"""Pure encounter rules (no database access) so they are easy to test and reason about."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable, Optional

from wildquest.observations.models import Observation

from .config import GamificationConfig

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = phi2 - phi1, math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def encounter_window(config: GamificationConfig) -> timedelta:
    return timedelta(hours=config.encounter_window_hours)


def is_same_encounter(earlier: Observation, observed_at: datetime, latitude: Optional[float],
                      longitude: Optional[float], config: GamificationConfig) -> bool:
    """Same user and species are assumed by the caller. Same encounter when the sightings
    are within the time window and, if both have a location, within the radius.

    When either sighting lacks a location the rule falls back to time only. This is the
    conservative choice: withholding GPS cannot be used to earn extra encounter XP.
    """
    if abs(observed_at - earlier.observed_at) > encounter_window(config):
        return False
    if earlier.latitude is None or latitude is None:
        return True
    distance = haversine_m(earlier.latitude, earlier.longitude, latitude, longitude)
    return distance <= config.encounter_radius_m


def matching_encounter(candidates: Iterable[Observation], observed_at: datetime,
                       latitude: Optional[float], longitude: Optional[float],
                       config: GamificationConfig) -> Optional[Observation]:
    """The closest-in-time earlier sighting that belongs to the same encounter, if any."""
    matches = [c for c in candidates
               if is_same_encounter(c, observed_at, latitude, longitude, config)]
    if not matches:
        return None
    return min(matches, key=lambda c: (abs(observed_at - c.observed_at), c.id))
