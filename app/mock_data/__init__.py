"""Deterministic mock providers so the UI runs without models or a database."""

import hashlib
import json
from pathlib import Path

from ..species import SPECIES

DATA_DIR = Path(__file__).resolve().parent
SCENARIOS = ("single", "multiple", "uncertain", "no_animal", "error")
# Scenarios picked automatically from the photo; no_animal/error are chosen explicitly.
AUTO_SCENARIOS = ("single", "multiple", "uncertain")
# A filename containing one of these words forces that scenario (handy for demos and tests).
FILENAME_HINTS = {"multi": "multiple", "uncertain": "uncertain", "noanimal": "no_animal",
                  "empty": "no_animal", "error": "error"}


def _load(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def choose_scenario(image_bytes: bytes, filename: str = "") -> str:
    """Same photo -> same scenario, so results are repeatable."""
    lowered = filename.lower()
    for hint, scenario in FILENAME_HINTS.items():
        if hint in lowered:
            return scenario
    digest = hashlib.sha256(image_bytes).digest()
    return AUTO_SCENARIOS[digest[0] % len(AUTO_SCENARIOS)]


class MockInferenceProvider:
    """Stands in for the detection + classification backend (issue #11)."""

    def __init__(self):
        self.responses = {k: v for k, v in _load("sample_responses.json").items()
                          if not k.startswith("_")}

    def predict(self, image_bytes: bytes, filename: str = "", scenario: str = None) -> dict:
        scenario = scenario or choose_scenario(image_bytes, filename)
        if scenario not in self.responses:
            raise KeyError(f"Unknown mock scenario {scenario!r}")
        return json.loads(json.dumps(self.responses[scenario]))  # independent copy


class MockObservationProvider:
    """Stands in for ObservationService (issue #12): same method names and output shapes."""

    def __init__(self):
        self.profile = _load("sample_profile.json")
        self.user_id = self.profile["user_id"]

    def history(self, user_id=None, limit=50):
        rows = []
        for raw in self.profile["history"][:limit]:
            row = dict(raw)
            row["user_id"] = self.user_id
            row["species"] = SPECIES[row["class_id"]].scientific_name
            rows.append(row)
        return rows

    def collection(self, user_id=None):
        species = []
        history = self.history(limit=None)
        for sp in SPECIES:
            mine = [h for h in history if h["class_id"] == sp.class_id]
            verified = [h for h in mine if h["status"] == "VERIFIED"]
            times = sorted(h["observed_at"] for h in verified)
            species.append({
                "class_id": sp.class_id,
                "species": sp.scientific_name,
                "discovered": bool(verified),
                "verified_observations": len(verified),
                "pending_review": sum(h["status"] == "NEEDS_REVIEW" for h in mine),
                "encounters": len({h["encounter_id"] for h in verified}),
                "first_seen": times[0] if times else None,
                "last_seen": times[-1] if times else None,
                "best_confidence": max((h["confidence"] for h in verified), default=None),
            })
        discovered = sum(s["discovered"] for s in species)
        return {"user_id": self.user_id, "discovered": discovered,
                "total_species": len(SPECIES), "progress": discovered / len(SPECIES),
                "species": species}

    def summary(self, user_id=None):
        history = self.history(limit=None)
        collection = self.collection()
        count = lambda status: sum(h["status"] == status for h in history)  # noqa: E731
        return {
            "user_id": self.user_id,
            "total_xp": self.profile["total_xp"],
            "discovered_species": collection["discovered"],
            "total_species": collection["total_species"],
            "observations": len(history),
            "verified": count("VERIFIED"),
            "needs_review": count("NEEDS_REVIEW"),
            "rejected": count("REJECTED"),
            "last_observed_at": history[0]["observed_at"] if history else None,
        }
    