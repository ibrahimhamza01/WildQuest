"""Observation records, validation of incoming prediction events, and result types."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "configs/class_mapping.json"
MAX_TEXT_LENGTH = 256


class ObservationError(ValueError):
    """An observation event or review request is invalid."""


class RequestConflictError(ObservationError):
    """A request_id was reused with a different payload."""


class ObservationNotFoundError(LookupError):
    """No observation exists with the given ID."""


class Status(str, Enum):
    VERIFIED = "VERIFIED"          # counted in the collection and eligible for XP
    NEEDS_REVIEW = "NEEDS_REVIEW"  # low confidence: stored, but no XP until approved
    REJECTED = "REJECTED"          # reviewed and rejected: never earns XP


class XPOutcome(str, Enum):
    FIRST_DISCOVERY = "FIRST_DISCOVERY"          # first verified sighting of a species
    NEW_ENCOUNTER = "NEW_ENCOUNTER"              # verified sighting outside any earlier encounter
    SAME_ENCOUNTER = "SAME_ENCOUNTER"            # repeat of an encounter already rewarded
    DAILY_LIMIT_REACHED = "DAILY_LIMIT_REACHED"  # new encounter, but the species' daily cap is used
    PENDING_REVIEW = "PENDING_REVIEW"            # stored as NEEDS_REVIEW, no XP yet
    REJECTED = "REJECTED"                        # reviewer rejected the observation
    DUPLICATE_REQUEST = "DUPLICATE_REQUEST"      # retried request: nothing new recorded or awarded


def load_species_names(mapping_path=DEFAULT_MAPPING_PATH) -> Tuple[str, ...]:
    """Scientific names in frozen class-ID order from class_mapping.json."""
    path = Path(mapping_path)
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))["id_to_label"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ObservationError(f"Cannot read class mapping {path}: {exc}") from exc
    ids = [str(i) for i in range(len(labels))]
    if not ids or set(labels) != set(ids):
        raise ObservationError(f"{path} must define contiguous class IDs starting at 0")
    return tuple(labels[i] for i in ids)


def to_utc_text(moment: datetime) -> str:
    """Fixed-width UTC ISO-8601 text, so string order equals time order in SQLite."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def from_utc_text(text: str) -> datetime:
    return datetime.fromisoformat(text)


def parse_timestamp(value: Any, field: str = "observed_at") -> datetime:
    """Accept a timezone-aware datetime or ISO-8601 string (``Z`` allowed); return UTC."""
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            raise ObservationError(f"{field} must be an ISO-8601 timestamp, got {value!r}") from None
    if not isinstance(value, datetime):
        raise ObservationError(f"{field} must be an ISO-8601 timestamp or datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ObservationError(
            f"{field} must include a timezone (for example '2026-11-20T14:05:00Z')")
    return value.astimezone(timezone.utc)


def _text(value: Any, field: str, required: bool) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ObservationError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ObservationError(f"{field} must be a string")
    value = value.strip()
    if len(value) > MAX_TEXT_LENGTH:
        raise ObservationError(f"{field} must be at most {MAX_TEXT_LENGTH} characters")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationError(f"{field} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ObservationError(f"{field} must be finite")
    return value


@dataclass(frozen=True)
class ObservationEvent:
    """One classified animal submitted for recording.

    ``request_id`` is chosen by the client (e.g. ``"<upload uuid>:<detection index>"``)
    and makes retries safe: the same request is never recorded or rewarded twice.
    """

    request_id: str
    user_id: str
    class_id: int
    confidence: float
    observed_at: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    image_ref: Optional[str] = None

    @property
    def has_location(self) -> bool:
        return self.latitude is not None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], num_classes: int) -> "ObservationEvent":
        if not isinstance(data, Mapping):
            raise ObservationError("Observation event must be a JSON object / mapping")
        class_id = data.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ObservationError(f"class_id must be an integer, got {class_id!r}")
        if not 0 <= class_id < num_classes:
            raise ObservationError(f"class_id must be in [0, {num_classes - 1}], got {class_id}")
        confidence = _number(data.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ObservationError(f"confidence must be in [0, 1], got {confidence}")

        latitude, longitude = data.get("latitude"), data.get("longitude")
        if (latitude is None) != (longitude is None):
            raise ObservationError("latitude and longitude must be given together or not at all")
        if latitude is not None:
            latitude, longitude = _number(latitude, "latitude"), _number(longitude, "longitude")
            if not -90.0 <= latitude <= 90.0:
                raise ObservationError(f"latitude must be in [-90, 90], got {latitude}")
            if not -180.0 <= longitude <= 180.0:
                raise ObservationError(f"longitude must be in [-180, 180], got {longitude}")

        return cls(
            request_id=_text(data.get("request_id"), "request_id", required=True),
            user_id=_text(data.get("user_id"), "user_id", required=True),
            class_id=class_id,
            confidence=confidence,
            observed_at=parse_timestamp(data.get("observed_at")),
            latitude=latitude,
            longitude=longitude,
            image_ref=_text(data.get("image_ref"), "image_ref", required=False),
        )


@dataclass(frozen=True)
class Observation:
    """A stored observation."""

    id: int
    request_id: str
    user_id: str
    class_id: int
    species: str
    confidence: float
    observed_at: datetime
    latitude: Optional[float]
    longitude: Optional[float]
    status: Status
    encounter_id: Optional[int]
    image_ref: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime]

    @property
    def has_location(self) -> bool:
        return self.latitude is not None

    def same_payload(self, event: ObservationEvent) -> bool:
        return (self.user_id, self.class_id, self.confidence, self.observed_at,
                self.latitude, self.longitude, self.image_ref) == (
            event.user_id, event.class_id, event.confidence, event.observed_at,
            event.latitude, event.longitude, event.image_ref)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for key in ("observed_at", "created_at", "reviewed_at"):
            if data[key] is not None:
                data[key] = to_utc_text(data[key])
        return data


@dataclass(frozen=True)
class RecordResult:
    """What happened to one submitted (or reviewed) observation."""

    observation: Observation
    outcome: XPOutcome
    xp_awarded: int
    total_xp: int
    new_species: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "outcome": self.outcome.value,
            "xp_awarded": self.xp_awarded,
            "total_xp": self.total_xp,
            "new_species": self.new_species,
        }
