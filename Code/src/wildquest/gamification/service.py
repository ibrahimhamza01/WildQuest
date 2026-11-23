"""The WildQuest observation service: record predictions, award XP, build the collection.

This is the interface the inference pipeline (issue #11) and the UI (issue #13) call.
It accepts plain dicts so results can come straight from JSON.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from wildquest.observations.models import (Observation, ObservationError, ObservationEvent,
                                           ObservationNotFoundError, RecordResult,
                                           RequestConflictError, Status, XPOutcome,
                                           load_species_names, to_utc_text)
from wildquest.observations.store import ObservationStore

from .config import GamificationConfig
from .rules import encounter_window, matching_encounter

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObservationService:
    """Record observations and award XP with encounter deduplication.

    Usage::

        with ObservationService("outputs/observations/wildquest.db") as service:
            result = service.record({
                "request_id": "upload-7f3c:0", "user_id": "ege", "class_id": 17,
                "confidence": 0.93, "observed_at": "2026-11-20T14:05:00Z",
                "latitude": 38.8996, "longitude": -77.0490,
            })
            print(result.outcome, result.xp_awarded, result.total_xp)
    """

    def __init__(self, db_path, config: Optional[GamificationConfig] = None,
                 class_names: Optional[Sequence[str]] = None, clock: Clock = utc_now):
        self.config = config if config is not None else GamificationConfig.load()
        self.class_names = tuple(class_names) if class_names is not None else load_species_names()
        self._clock = clock
        self.store = ObservationStore(db_path, self.class_names)

    def __enter__(self) -> "ObservationService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)

    # ----- writes -------------------------------------------------------------------

    def record(self, event: Union[ObservationEvent, Mapping[str, Any]]) -> RecordResult:
        """Store one classified animal and award XP if it qualifies.

        * confidence < review_threshold  -> NEEDS_REVIEW, no XP until ``review`` approves it
        * a retried request_id with the same payload -> DUPLICATE_REQUEST, nothing awarded
        * a reused request_id with a different payload -> RequestConflictError
        """
        if not isinstance(event, ObservationEvent):
            event = ObservationEvent.from_dict(event, len(self.class_names))
        if not 0 <= event.class_id < len(self.class_names):
            raise ObservationError(f"class_id must be in [0, {len(self.class_names) - 1}]")
        now = self._now()
        if event.observed_at > now + timedelta(minutes=self.config.max_future_skew_minutes):
            raise ObservationError(
                f"observed_at {to_utc_text(event.observed_at)} is in the future")

        with self.store.transaction():
            existing = self.store.get_by_request_id(event.request_id)
            if existing is not None:
                if not existing.same_payload(event):
                    raise RequestConflictError(
                        f"request_id {event.request_id!r} was already used for a different "
                        "observation")
                return self._result(existing, XPOutcome.DUPLICATE_REQUEST, 0, False)

            if event.confidence >= self.config.review_threshold:
                observation = self.store.insert_observation(event, Status.VERIFIED, now)
                outcome, points = self._apply_verified(observation, now)
            else:
                observation = self.store.insert_observation(event, Status.NEEDS_REVIEW, now)
                outcome, points = XPOutcome.PENDING_REVIEW, 0
            return self._result(self.store.get(observation.id), outcome, points,
                                outcome is XPOutcome.FIRST_DISCOVERY)

    def review(self, observation_id: int, approve: bool) -> RecordResult:
        """Approve or reject a NEEDS_REVIEW observation. Approval awards XP as if it had
        been verified now. Repeating the same decision is a no-op; reversing one is an error."""
        now = self._now()
        target = Status.VERIFIED if approve else Status.REJECTED
        with self.store.transaction():
            observation = self.store.get(observation_id)
            if observation is None:
                raise ObservationNotFoundError(f"No observation with id {observation_id}")
            if observation.status is not Status.NEEDS_REVIEW:
                if observation.status is target:
                    return self._result(observation, XPOutcome.DUPLICATE_REQUEST, 0, False)
                raise ObservationError(
                    f"Observation {observation_id} is already {observation.status.value}")
            self.store.update_status(observation_id, target, now)
            observation = self.store.get(observation_id)
            if not approve:
                return self._result(observation, XPOutcome.REJECTED, 0, False)
            outcome, points = self._apply_verified(observation, now)
            return self._result(self.store.get(observation_id), outcome, points,
                                outcome is XPOutcome.FIRST_DISCOVERY)

    def _apply_verified(self, observation: Observation, now: datetime) -> Tuple[XPOutcome, int]:
        """Assign the encounter and write at most one ledger entry. Runs inside a transaction."""
        window = encounter_window(self.config)
        candidates = self.store.verified_between(
            observation.user_id, observation.class_id, observation.observed_at - window,
            observation.observed_at + window, exclude_id=observation.id)
        match = matching_encounter(candidates, observation.observed_at, observation.latitude,
                                   observation.longitude, self.config)
        if match is not None:
            self.store.set_encounter(observation.id, match.encounter_id or match.id)
            return XPOutcome.SAME_ENCOUNTER, 0

        self.store.set_encounter(observation.id, observation.id)
        if not self.store.has_first_discovery(observation.user_id, observation.class_id):
            outcome, points = XPOutcome.FIRST_DISCOVERY, self.config.first_discovery_xp
        else:
            # Daily cap uses the server clock: observed_at comes from the client and could be
            # back-dated to spread one animal over many fake "encounters".
            day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
            used = self.store.count_awards(observation.user_id, observation.class_id,
                                           XPOutcome.NEW_ENCOUNTER, day_start,
                                           day_start + timedelta(days=1))
            if used >= self.config.max_encounter_awards_per_species_per_day:
                return XPOutcome.DAILY_LIMIT_REACHED, 0
            outcome, points = XPOutcome.NEW_ENCOUNTER, self.config.encounter_xp
        if points > 0:
            self.store.insert_award(observation, outcome, points, now)
        return outcome, points

    def _result(self, observation: Observation, outcome: XPOutcome, points: int,
                new_species: bool) -> RecordResult:
        return RecordResult(observation=observation, outcome=outcome, xp_awarded=points,
                            total_xp=self.store.total_xp(observation.user_id),
                            new_species=new_species)

    # ----- reads --------------------------------------------------------------------

    def get(self, observation_id: int) -> Observation:
        observation = self.store.get(observation_id)
        if observation is None:
            raise ObservationNotFoundError(f"No observation with id {observation_id}")
        return observation

    def history(self, user_id: str, limit: Optional[int] = 50,
                status: Optional[Status] = None) -> List[Observation]:
        """A user's observations, newest first."""
        return self.store.history(user_id, limit, status)

    def pending_review(self, user_id: str) -> List[Observation]:
        return self.store.history(user_id, None, Status.NEEDS_REVIEW)

    def total_xp(self, user_id: str) -> int:
        return self.store.total_xp(user_id)

    def xp_history(self, user_id: str) -> List[dict]:
        return self.store.awards(user_id)

    def collection(self, user_id: str) -> Dict[str, Any]:
        """One entry per species (all 25), with discovery state and counts."""
        stats = {row["class_id"]: row for row in self.store.species_stats(user_id)}
        species = []
        for class_id, name in enumerate(self.class_names):
            row = stats.get(class_id)
            verified = row["verified"] if row else 0
            species.append({
                "class_id": class_id,
                "species": name,
                "discovered": bool(verified),
                "verified_observations": verified,
                "pending_review": row["pending"] if row else 0,
                "encounters": row["encounters"] if row else 0,
                "first_seen": row["first_seen"] if row and verified else None,
                "last_seen": row["last_seen"] if row and verified else None,
                "best_confidence": row["best_confidence"] if row and verified else None,
            })
        discovered = sum(entry["discovered"] for entry in species)
        return {
            "user_id": user_id,
            "discovered": discovered,
            "total_species": len(self.class_names),
            "progress": discovered / len(self.class_names),
            "species": species,
        }

    def summary(self, user_id: str) -> Dict[str, Any]:
        """Profile numbers for the UI header."""
        collection = self.collection(user_id)
        counts = self.store.status_counts(user_id)
        latest = self.store.history(user_id, 1)
        return {
            "user_id": user_id,
            "total_xp": self.store.total_xp(user_id),
            "discovered_species": collection["discovered"],
            "total_species": collection["total_species"],
            "observations": sum(counts.values()),
            "verified": counts[Status.VERIFIED.value],
            "needs_review": counts[Status.NEEDS_REVIEW.value],
            "rejected": counts[Status.REJECTED.value],
            "last_observed_at": to_utc_text(latest[0].observed_at) if latest else None,
        }

