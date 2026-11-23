"""Observation storage tests (issue #12): validation, persistence, history, collection."""

import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from wildquest.gamification import GamificationConfig, ObservationService  # noqa: E402
from wildquest.observations import (ObservationError, ObservationEvent,  # noqa: E402
                                    ObservationNotFoundError, ObservationStore, Status,
                                    load_species_names, parse_timestamp)

NOW = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
NAMES = load_species_names()


def event(request_id="r1", user_id="ege", class_id=17, confidence=0.95,
          observed_at="2026-11-20T14:00:00Z", **extra):
    return {"request_id": request_id, "user_id": user_id, "class_id": class_id,
            "confidence": confidence, "observed_at": observed_at, **extra}


class EventValidationTests(unittest.TestCase):
    def parse(self, **changes):
        return ObservationEvent.from_dict(event(**changes), 25)

    def test_valid_event_is_normalized_to_utc(self):
        parsed = self.parse(observed_at="2026-11-20T09:00:00-05:00",
                            latitude=38.9, longitude=-77.05, image_ref=" uploads/a.jpg ")
        self.assertEqual(parsed.observed_at, datetime(2026, 11, 20, 14, tzinfo=timezone.utc))
        self.assertEqual(parsed.image_ref, "uploads/a.jpg")
        self.assertTrue(parsed.has_location)

    def test_missing_location_is_allowed(self):
        parsed = self.parse()
        self.assertIsNone(parsed.latitude)
        self.assertFalse(parsed.has_location)

    def test_rejections(self):
        cases = {
            "class_id must be an integer": {"class_id": "17"},
            "class_id must be in": {"class_id": 25},
            "class_id must be an integer, got True": {"class_id": True},
            "confidence must be in": {"confidence": 1.5},
            "confidence must be a number": {"confidence": "high"},
            "confidence must be finite": {"confidence": float("nan")},
            "user_id is required": {"user_id": "  "},
            "request_id is required": {"request_id": None},
            "timezone": {"observed_at": "2026-11-20T14:00:00"},
            "ISO-8601": {"observed_at": "yesterday"},
            "given together": {"latitude": 38.9},
            "latitude must be in": {"latitude": 91, "longitude": 0},
            "longitude must be in": {"latitude": 0, "longitude": -181},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ObservationError, message):
                    self.parse(**changes)

    def test_non_mapping_rejected(self):
        with self.assertRaises(ObservationError):
            ObservationEvent.from_dict(["not", "a", "dict"], 25)

    def test_parse_timestamp_accepts_datetime(self):
        self.assertEqual(parse_timestamp(NOW), NOW)

    def test_species_names_follow_frozen_mapping(self):
        self.assertEqual(len(NAMES), 25)
        self.assertEqual(NAMES[17], "Procyon lotor")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "nested" / "wildquest.db"

    def tearDown(self):
        self.tmp.cleanup()

    def service(self):
        return ObservationService(self.db, GamificationConfig(), NAMES, clock=lambda: NOW)

    def test_observations_and_xp_survive_reopening(self):
        with self.service() as service:
            first = service.record(event(latitude=38.9, longitude=-77.05))
        with self.service() as service:
            stored = service.get(first.observation.id)
            self.assertEqual(stored.species, "Procyon lotor")
            self.assertEqual((stored.latitude, stored.longitude), (38.9, -77.05))
            self.assertEqual(stored.status, Status.VERIFIED)
            self.assertEqual(service.total_xp("ege"), 100)
            self.assertEqual(service.collection("ege")["discovered"], 1)

    def test_newer_schema_is_refused(self):
        self.service().close()
        with sqlite3.connect(self.db) as raw:
            raw.execute("PRAGMA user_version = 99")
        with self.assertRaisesRegex(ObservationError, "schema v99"):
            ObservationStore(self.db, NAMES)

    def test_database_constraints_back_up_validation(self):
        self.service().close()
        with sqlite3.connect(self.db) as raw:
            with self.assertRaises(sqlite3.IntegrityError):
                raw.execute("INSERT INTO observations (request_id, user_id, class_id, confidence,"
                            " observed_at, status, created_at) VALUES"
                            " ('x', 'u', 1, 0.9, 't', 'MAYBE', 't')")

    def test_missing_observation(self):
        with self.service() as service:
            with self.assertRaises(ObservationNotFoundError):
                service.get(123)


class HistoryAndCollectionTests(unittest.TestCase):
    def setUp(self):
        self.service = ObservationService(":memory:", GamificationConfig(), NAMES,
                                          clock=lambda: NOW)

    def tearDown(self):
        self.service.close()

    def test_history_is_newest_first_per_user_and_filterable(self):
        self.service.record(event("a", observed_at="2026-11-20T08:00:00Z"))
        self.service.record(event("b", class_id=4, observed_at="2026-11-20T12:00:00Z"))
        self.service.record(event("c", class_id=5, confidence=0.3,
                                  observed_at="2026-11-20T10:00:00Z"))
        self.service.record(event("d", user_id="alex"))
        history = self.service.history("ege")
        self.assertEqual([o.request_id for o in history], ["b", "c", "a"])
        self.assertEqual([o.request_id for o in self.service.history("ege", limit=1)], ["b"])
        self.assertEqual([o.request_id for o in self.service.pending_review("ege")], ["c"])
        self.assertEqual(history[0].to_dict()["observed_at"], "2026-11-20T12:00:00.000000+00:00")

    def test_collection_lists_all_species_and_ignores_unverified(self):
        self.service.record(event("a", class_id=17, confidence=0.9,
                                  observed_at="2026-11-20T08:00:00Z"))
        self.service.record(event("b", class_id=17, confidence=0.97,
                                  observed_at="2026-11-20T15:00:00Z"))
        self.service.record(event("c", class_id=3, confidence=0.4))
        collection = self.service.collection("ege")
        self.assertEqual(len(collection["species"]), 25)
        self.assertEqual(collection["discovered"], 1)
        self.assertAlmostEqual(collection["progress"], 1 / 25)
        raccoon = collection["species"][17]
        self.assertTrue(raccoon["discovered"])
        self.assertEqual(raccoon["verified_observations"], 2)
        self.assertEqual(raccoon["encounters"], 2)
        self.assertEqual(raccoon["best_confidence"], 0.97)
        self.assertTrue(raccoon["first_seen"].startswith("2026-11-20T08:00"))
        hawk = collection["species"][3]
        self.assertFalse(hawk["discovered"])
        self.assertEqual(hawk["pending_review"], 1)
        self.assertIsNone(hawk["first_seen"])

    def test_summary(self):
        self.service.record(event("a"))
        self.service.record(event("b", class_id=3, confidence=0.4))
        summary = self.service.summary("ege")
        self.assertEqual(summary["total_xp"], 100)
        self.assertEqual(summary["discovered_species"], 1)
        self.assertEqual((summary["observations"], summary["verified"],
                          summary["needs_review"], summary["rejected"]), (2, 1, 1, 0))
        empty = self.service.summary("nobody")
        self.assertEqual((empty["total_xp"], empty["observations"]), (0, 0))
        self.assertIsNone(empty["last_observed_at"])

    def test_future_timestamp_rejected(self):
        too_late = (NOW + timedelta(hours=1)).isoformat()
        with self.assertRaisesRegex(ObservationError, "future"):
            self.service.record(event(observed_at=too_late))
        # Small clock skew is tolerated.
        self.service.record(event("ok", observed_at=(NOW + timedelta(minutes=5)).isoformat()))


if __name__ == "__main__":
    unittest.main()
