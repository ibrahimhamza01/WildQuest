"""XP and encounter-deduplication tests (issue #12) using synthetic observation events."""

import json
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from wildquest.gamification import (DEFAULT_CONFIG_PATH, GamificationConfig,  # noqa: E402
                                    ObservationService, haversine_m)
from wildquest.observations import (ObservationError, RequestConflictError,  # noqa: E402
                                    Status, XPOutcome, load_species_names)

START = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)
NAMES = load_species_names()
# Two points in Rock Creek Park about 1.1 km apart.
PARK_A = {"latitude": 38.9580, "longitude": -77.0430}
PARK_B = {"latitude": 38.9680, "longitude": -77.0430}
PARK_A_NEARBY = {"latitude": 38.9585, "longitude": -77.0430}  # ~56 m from PARK_A


class Clock:
    def __init__(self, now=START + timedelta(hours=12)):
        self.now = now

    def __call__(self):
        return self.now


def event(request_id, hours=0.0, class_id=13, confidence=0.9, user_id="ege", **extra):
    return {"request_id": request_id, "user_id": user_id, "class_id": class_id,
            "confidence": confidence,
            "observed_at": (START + timedelta(hours=hours)).isoformat(), **extra}


class ServiceTestCase(unittest.TestCase):
    config = GamificationConfig()

    def setUp(self):
        self.clock = Clock()
        self.service = ObservationService(":memory:", self.config, NAMES, clock=self.clock)

    def tearDown(self):
        self.service.close()

    def record(self, *args, **kwargs):
        return self.service.record(event(*args, **kwargs))


class DiscoveryAndEncounterTests(ServiceTestCase):
    def test_first_discovery_awards_bonus_once_per_species(self):
        first = self.record("a", **PARK_A)
        self.assertEqual(first.outcome, XPOutcome.FIRST_DISCOVERY)
        self.assertEqual((first.xp_awarded, first.total_xp), (100, 100))
        self.assertTrue(first.new_species)
        other_species = self.record("b", class_id=17, **PARK_A)
        self.assertEqual(other_species.outcome, XPOutcome.FIRST_DISCOVERY)
        self.assertEqual(other_species.total_xp, 200)

    def test_repeats_in_same_encounter_award_nothing(self):
        self.record("a", **PARK_A)
        for i, hours in enumerate((0.1, 1, 3, 5.9)):
            result = self.record(f"repeat-{i}", hours=hours, **PARK_A_NEARBY)
            self.assertEqual(result.outcome, XPOutcome.SAME_ENCOUNTER)
            self.assertEqual(result.xp_awarded, 0)
        self.assertEqual(self.service.total_xp("ege"), 100)

    def test_same_encounter_shares_encounter_id(self):
        first = self.record("a", **PARK_A)
        repeat = self.record("b", hours=1, **PARK_A)
        self.assertEqual(first.observation.encounter_id, first.observation.id)
        self.assertEqual(repeat.observation.encounter_id, first.observation.id)

    def test_new_encounter_after_time_window(self):
        self.record("a", **PARK_A)
        later = self.record("b", hours=6.5, **PARK_A)
        self.assertEqual(later.outcome, XPOutcome.NEW_ENCOUNTER)
        self.assertEqual(later.xp_awarded, 10)
        self.assertFalse(later.new_species)

    def test_new_encounter_beyond_radius(self):
        self.record("a", **PARK_A)
        far = self.record("b", hours=1, **PARK_B)
        self.assertGreater(haversine_m(**_pair(PARK_A, PARK_B)), 500)
        self.assertEqual(far.outcome, XPOutcome.NEW_ENCOUNTER)

    def test_missing_location_falls_back_to_time_only(self):
        self.record("a", **PARK_A)
        # No GPS within the window: treated as the same encounter (cannot farm by hiding GPS).
        self.assertEqual(self.record("b", hours=1).outcome, XPOutcome.SAME_ENCOUNTER)
        self.assertEqual(self.record("c", hours=2).outcome, XPOutcome.SAME_ENCOUNTER)
        # Outside the window it is a new encounter even without a location.
        self.assertEqual(self.record("d", hours=9).outcome, XPOutcome.NEW_ENCOUNTER)

    def test_out_of_order_submission_matches_later_sighting(self):
        self.record("late", hours=5, **PARK_A)
        early = self.record("early", hours=1, **PARK_A)  # uploaded afterwards, taken earlier
        self.assertEqual(early.outcome, XPOutcome.SAME_ENCOUNTER)

    def test_users_are_independent(self):
        self.record("a", user_id="ege", **PARK_A)
        other = self.record("b", user_id="alex", **PARK_A)
        self.assertEqual(other.outcome, XPOutcome.FIRST_DISCOVERY)
        self.assertEqual(self.service.total_xp("ege"), 100)
        self.assertEqual(self.service.total_xp("alex"), 100)

    def test_daily_cap_on_new_encounters_uses_server_day(self):
        self.clock.now = START + timedelta(hours=40)  # all uploads happen on one server day
        self.record("first", **PARK_A)
        outcomes = [self.record(f"e{i}", hours=7 * (i + 1), **PARK_A).outcome for i in range(5)]
        self.assertEqual(outcomes, [XPOutcome.NEW_ENCOUNTER] * 3
                         + [XPOutcome.DAILY_LIMIT_REACHED] * 2)
        self.assertEqual(self.service.total_xp("ege"), 100 + 3 * 10)
        # Back-dating does not reset the cap: it counts awards by server time.
        backdated = self.record("old", hours=-100, **PARK_A)
        self.assertEqual(backdated.outcome, XPOutcome.DAILY_LIMIT_REACHED)
        # The next server day allows new encounters again.
        self.clock.now += timedelta(days=1)
        self.assertEqual(self.record("next-day", hours=60, **PARK_A).outcome,
                         XPOutcome.NEW_ENCOUNTER)

    def test_xp_history_matches_total(self):
        self.record("a", **PARK_A)
        self.record("b", hours=7, **PARK_A)
        self.record("c", hours=7.5, **PARK_A)
        awards = self.service.xp_history("ege")
        self.assertEqual([(a["reason"], a["points"]) for a in awards],
                         [("FIRST_DISCOVERY", 100), ("NEW_ENCOUNTER", 10)])
        self.assertEqual(sum(a["points"] for a in awards), self.service.total_xp("ege"))
        self.assertEqual(awards[0]["species"], NAMES[13])


class DuplicateRequestTests(ServiceTestCase):
    def test_retried_request_is_not_recorded_or_awarded_twice(self):
        first = self.record("upload-1:0", **PARK_A)
        retry = self.record("upload-1:0", **PARK_A)
        self.assertEqual(retry.outcome, XPOutcome.DUPLICATE_REQUEST)
        self.assertEqual(retry.xp_awarded, 0)
        self.assertEqual(retry.observation.id, first.observation.id)
        self.assertEqual(retry.total_xp, 100)
        self.assertEqual(len(self.service.history("ege")), 1)

    def test_reused_request_id_with_different_payload_is_refused(self):
        self.record("upload-1:0", **PARK_A)
        with self.assertRaises(RequestConflictError):
            self.record("upload-1:0", class_id=4, **PARK_A)
        self.assertEqual(len(self.service.history("ege")), 1)

    def test_concurrent_duplicate_requests_award_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "wildquest.db"
            ObservationService(db, self.config, NAMES, clock=self.clock).close()
            results, errors = [], []

            def submit():
                try:
                    with ObservationService(db, self.config, NAMES, clock=self.clock) as s:
                        results.append(s.record(event("same", **PARK_A)).outcome)
                except Exception as exc:  # pragma: no cover - reported below
                    errors.append(exc)

            threads = [threading.Thread(target=submit) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(results.count(XPOutcome.FIRST_DISCOVERY), 1)
            self.assertEqual(results.count(XPOutcome.DUPLICATE_REQUEST), 7)
            with ObservationService(db, self.config, NAMES, clock=self.clock) as s:
                self.assertEqual(s.total_xp("ege"), 100)


class ReviewTests(ServiceTestCase):
    def test_low_confidence_needs_review_and_awards_nothing(self):
        result = self.record("a", confidence=0.55, **PARK_A)
        self.assertEqual(result.observation.status, Status.NEEDS_REVIEW)
        self.assertEqual(result.outcome, XPOutcome.PENDING_REVIEW)
        self.assertEqual(result.total_xp, 0)
        self.assertFalse(self.service.collection("ege")["species"][13]["discovered"])

    def test_threshold_boundary_is_inclusive(self):
        self.assertEqual(self.record("a", confidence=0.70).observation.status, Status.VERIFIED)
        self.assertEqual(self.record("b", class_id=1, confidence=0.6999).observation.status,
                         Status.NEEDS_REVIEW)

    def test_approval_awards_xp_once(self):
        pending = self.record("a", confidence=0.4, **PARK_A)
        approved = self.service.review(pending.observation.id, approve=True)
        self.assertEqual(approved.observation.status, Status.VERIFIED)
        self.assertIsNotNone(approved.observation.reviewed_at)
        self.assertEqual(approved.outcome, XPOutcome.FIRST_DISCOVERY)
        self.assertEqual(approved.total_xp, 100)
        again = self.service.review(pending.observation.id, approve=True)
        self.assertEqual((again.outcome, again.xp_awarded, again.total_xp),
                         (XPOutcome.DUPLICATE_REQUEST, 0, 100))

    def test_approval_still_applies_encounter_rules(self):
        self.record("a", **PARK_A)
        pending = self.record("b", hours=1, confidence=0.4, **PARK_A)
        approved = self.service.review(pending.observation.id, approve=True)
        self.assertEqual(approved.outcome, XPOutcome.SAME_ENCOUNTER)
        self.assertEqual(self.service.total_xp("ege"), 100)

    def test_rejection_and_reversal(self):
        pending = self.record("a", confidence=0.4)
        rejected = self.service.review(pending.observation.id, approve=False)
        self.assertEqual(rejected.observation.status, Status.REJECTED)
        self.assertEqual(rejected.outcome, XPOutcome.REJECTED)
        with self.assertRaisesRegex(ObservationError, "already REJECTED"):
            self.service.review(pending.observation.id, approve=True)
        self.assertEqual(self.service.total_xp("ege"), 0)

    def test_auto_verified_cannot_be_rejected(self):
        verified = self.record("a")
        with self.assertRaisesRegex(ObservationError, "already VERIFIED"):
            self.service.review(verified.observation.id, approve=False)


class StrictConfigTests(ServiceTestCase):
    config = GamificationConfig(review_threshold=0.9, first_discovery_xp=50, encounter_xp=5,
                                encounter_window_hours=1, encounter_radius_m=50,
                                max_encounter_awards_per_species_per_day=1)

    def test_custom_rules_are_used(self):
        self.assertEqual(self.record("low", confidence=0.85).outcome, XPOutcome.PENDING_REVIEW)
        self.assertEqual(self.record("a", **PARK_A).xp_awarded, 50)
        # 56 m apart: outside a 50 m radius -> new encounter worth 5.
        self.assertEqual(self.record("b", hours=0.5, **PARK_A_NEARBY).xp_awarded, 5)
        self.assertEqual(self.record("c", hours=2, **PARK_A).outcome,
                         XPOutcome.DAILY_LIMIT_REACHED)


class ConfigAndGeometryTests(unittest.TestCase):
    def test_repository_config_loads(self):
        config = GamificationConfig.load(DEFAULT_CONFIG_PATH)
        self.assertEqual(config, GamificationConfig())

    def test_invalid_config_rejected(self):
        for bad in ({"review_threshold": 1.5}, {"encounter_xp": -1}, {"encounter_xp": 2.5},
                    {"encounter_radius_m": float("nan")}, {"bonus": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GamificationConfig.from_dict(bad)

    def test_config_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps({"encounter_xp": 25}), encoding="utf-8")
            self.assertEqual(GamificationConfig.load(path).encounter_xp, 25)

    def test_haversine(self):
        self.assertAlmostEqual(haversine_m(0, 0, 0, 0), 0.0)
        self.assertAlmostEqual(haversine_m(**_pair(PARK_A, PARK_A_NEARBY)), 55.6, delta=0.5)
        # One degree of latitude is about 111.2 km.
        self.assertAlmostEqual(haversine_m(38, -77, 39, -77), 111_195, delta=50)


def _pair(a, b):
    return {"lat1": a["latitude"], "lon1": a["longitude"],
            "lat2": b["latitude"], "lon2": b["longitude"]}


if __name__ == "__main__":
    unittest.main()
