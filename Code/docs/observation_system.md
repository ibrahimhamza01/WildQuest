# Observation Storage, XP and Encounter Deduplication

The gamification backend records each classified animal as an observation, builds the user's 25-species collection, and awards XP without allowing repeat photos of the same animal to farm points. It uses only the Python standard library (`sqlite3`). It needs no model, detector, UI, or network service.

| Package | Contents |
|---|---|
| `wildquest.observations` | `ObservationEvent` (input validation), `Observation`, `RecordResult`, `Status`, `XPOutcome`, `ObservationStore` (SQLite) |
| `wildquest.gamification` | `GamificationConfig`, pure encounter rules (`is_same_encounter`, `haversine_m`), `ObservationService` (the API) |

Rules are configured in `Code/configs/gamification.json`.

## Quick start

Run from the repository root with `Code/src` on `PYTHONPATH`:

```python
from wildquest.gamification import ObservationService

with ObservationService("outputs/observations/wildquest.db") as service:
    result = service.record({
        "request_id": "upload-7f3c:0",      # client-generated, unique per detection
        "user_id": "ege",
        "class_id": 17,                      # Code/configs/class_mapping.json
        "confidence": 0.93,                  # classifier probability of class_id
        "observed_at": "2026-11-20T14:05:00Z",
        "latitude": 38.8996, "longitude": -77.0490,   # optional, both or neither
        "image_ref": "uploads/7f3c.jpg",               # optional
    })
    print(result.to_dict())   # outcome, xp_awarded, total_xp, new_species, observation
    print(service.summary("ege"))
    print(service.collection("ege")["discovered"])
```

The database file and its folder are created on first use. `outputs/` is ignored by Git. Use `":memory:"` for throwaway tests.

## Input event

| Field | Required | Rules |
|---|---|---|
| `request_id` | yes | Client-chosen idempotency key, ≤ 256 chars. For a photo with several animals use `"<upload id>:<detection index>"` |
| `user_id` | yes | Non-empty string (user or session ID) |
| `class_id` | yes | Integer 0–24 |
| `confidence` | yes | Number in [0, 1] |
| `observed_at` | yes | ISO-8601 with a timezone (`Z` or `±HH:MM`); stored in UTC. More than 10 minutes in the future is rejected |
| `latitude`, `longitude` | no | Both or neither; [-90, 90] and [-180, 180] |
| `image_ref` | no | Free text, e.g. a stored image path |

Invalid events raise `ObservationError` with a message naming the field.

Each detection from the inference pipeline (issue #11) becomes one event: `species_class_id` → `class_id`, classification confidence → `confidence`. Detector confidence and bounding boxes are not stored here.

## Rules

### Verification status

| Status | When | XP |
|---|---|---|
| `VERIFIED` | `confidence >= review_threshold` (default 0.70), or approved by `review()` | Eligible |
| `NEEDS_REVIEW` | `confidence < review_threshold` | None until approved |
| `REJECTED` | Rejected by `review()` | Never |

`service.review(observation_id, approve=True|False)` decides a `NEEDS_REVIEW` observation. Approval applies the normal XP rules at review time. Repeating the same decision is a no-op. Reversing a decision, including rejecting an auto-verified observation, raises `ObservationError`. Awarded XP is never revoked.

The 0.70 threshold is provisional. Issue #14 will recommend a value from validation data. Change it in `gamification.json`.

### Encounters

An **encounter** is one real-world sighting of an animal. A new verified observation joins an existing encounter when the same user has a verified observation of the same species:

- taken within `encounter_window_hours` (default 6 h) before or after it, **and**
- within `encounter_radius_m` (default 500 m), when both observations have a location.

If either observation has no location, only the time rule applies. Leaving out GPS therefore cannot earn extra XP.

Observations in the same encounter share `encounter_id`, the ID of the encounter's first observation. Matching works in both directions in time, so photos uploaded out of order are handled.

### XP

Each verified observation gets one outcome:

| Outcome | When | XP (default) |
|---|---|---|
| `FIRST_DISCOVERY` | First verified observation of this species for this user | 100 |
| `NEW_ENCOUNTER` | Distinct encounter of an already discovered species, with any positive award within the daily cap | 10 |
| `SAME_ENCOUNTER` | Part of an existing encounter | 0 |
| `DAILY_LIMIT_REACHED` | A positive encounter award would exceed the species' daily cap (default 3 awards) | 0 |

The other outcomes are `PENDING_REVIEW`, `REJECTED`, and `DUPLICATE_REQUEST`.

A user's first verified observation of a species marks it as discovered, whether verified automatically or approved through review. With `first_discovery_xp = 0`, it still returns `FIRST_DISCOVERY` with `new_species = True` and appears in the collection. The XP ledger records only positive awards.

The daily cap counts positive `NEW_ENCOUNTER` awards by **server time** (UTC day). Photo timestamps come from the client and could be back-dated to spread one animal across many fake encounters. With `encounter_xp = 0`, distinct encounters of a discovered species return `NEW_ENCOUNTER` with zero XP and use none of the award quota.

### Duplicate requests

- **Retried `request_id` with the same payload:** returns `DUPLICATE_REQUEST` with the original observation and `xp_awarded = 0`. Nothing new is stored, so clients can safely retry after timeouts.
- **Reused `request_id` with a different payload:** raises `RequestConflictError`.
- **Database guarantees:**
  - every check-and-award sequence runs in a `BEGIN IMMEDIATE` transaction, so concurrent processes cannot award twice
  - `xp_awards.observation_id` is `UNIQUE`
  - a partial unique index allows only one `FIRST_DISCOVERY` award per user and species

## Reading data

| Method | Returns |
|---|---|
| `history(user_id, limit=50, status=None)` | Observations, newest first |
| `pending_review(user_id)` | `NEEDS_REVIEW` observations |
| `collection(user_id)` | All 25 species: `discovered`, `verified_observations`, `pending_review`, `encounters`, `first_seen`, `last_seen`, `best_confidence`, plus overall `discovered` / `total_species` / `progress` |
| `summary(user_id)` | `total_xp`, discovered count, observation counts by status, `last_observed_at` |
| `total_xp(user_id)`, `xp_history(user_id)` | XP total and the ledger entries behind it |
| `get(observation_id)` | One observation (raises `ObservationNotFoundError`) |

`Observation.to_dict()` and `RecordResult.to_dict()` return JSON-ready dicts for the UI.

A species counts as discovered only after a verified observation. Pending observations appear in `pending_review`.

## Storage

The SQLite schema is versioned with `PRAGMA user_version` (currently 1). Opening a database with a newer schema fails instead of corrupting it.

| Table | Contents |
|---|---|
| `observations` | One row per event: IDs, class, confidence, UTC timestamps, optional location, status, `encounter_id`, `created_at` / `reviewed_at` |
| `xp_awards` | Append-only ledger: `observation_id`, `reason`, `points`, `awarded_at`. Total XP is the sum of this table |

CHECK constraints repeat the main validation rules in the database.

## Configuration

`Code/configs/gamification.json`:

| Setting | Default | Meaning |
|---|---|---|
| `review_threshold` | 0.70 | Minimum confidence for automatic `VERIFIED` |
| `first_discovery_xp` | 100 | XP for a new species |
| `encounter_xp` | 10 | XP for a new encounter of a known species |
| `encounter_window_hours` | 6 | Time window for one encounter |
| `encounter_radius_m` | 500 | Distance for one encounter, when locations are known |
| `max_encounter_awards_per_species_per_day` | 3 | Daily `NEW_ENCOUNTER` cap per species |
| `max_future_skew_minutes` | 10 | Allowed client clock skew |

Unknown keys and invalid values are rejected. Pass `GamificationConfig(...)` to the service to override settings in code.

## Out of scope

- Identifying individual animals. Encounters are a time and distance heuristic, not re-identification.
- Authentication: `user_id` is trusted as given.
- Revoking XP.
- Levels and badges.

## Tests

```bash
python -m unittest Code/tests/test_observations.py Code/tests/test_gamification.py -v
```

The tests use synthetic events and a fake clock. They cover:

- validation and missing locations
- persistence across reopening and schema-version protection
- history and the collection
- first discovery, encounter time and radius rules, the GPS-less fallback, and out-of-order uploads
- the daily cap against back-dating
- review flows
- idempotent and conflicting requests, including 8 concurrent duplicate submissions
- configuration validation
