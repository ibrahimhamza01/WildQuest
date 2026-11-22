"""Read the frozen species mapping and validate prediction CSV records."""

import csv
import json
import re
from decimal import Context, Decimal, InvalidOperation, localcontext
from pathlib import Path

MAPPING_PATH = Path(__file__).resolve().parents[3] / "configs/class_mapping.json"
PROBABILITY_SUM_TOLERANCE = Decimal("0.00001")
PROBABILITY_SUM_CONTEXT = Context(prec=28)
REQUIRED_COLUMNS = ("true_class_id", "pred_class_id")


def load_species():
    """Return scientific names in frozen class-ID order."""
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    labels = mapping.get("id_to_label") if isinstance(mapping, dict) else None
    ids = [str(i) for i in range(25)]
    if not isinstance(labels, dict) or labels.keys() != set(ids):
        raise ValueError(f"{MAPPING_PATH} must define class IDs 0 through 24")
    if any(not isinstance(name, str) or not name.strip() for name in labels.values()):
        raise ValueError(f"{MAPPING_PATH} must define a scientific name for each class")
    return tuple(labels[class_id] for class_id in ids)


def _class_id(value, column, location, class_count):
    value = value.strip()
    if not value:
        raise ValueError(f"{location}: missing value for {column}")
    try:
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise ValueError
        class_id = int(value)
    except ValueError:
        raise ValueError(f"{location}: {column} must be an integer class ID, got {value!r}") from None
    if not 0 <= class_id < class_count:
        raise ValueError(f"{location}: {column} must be in [0, {class_count - 1}], got {class_id}")
    return class_id


def _probability(value, column, location):
    value = value.strip()
    if not value:
        raise ValueError(f"{location}: missing value for {column}")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{location}: {column} must be a number, got {value!r}") from None
    if not number.is_finite():
        raise ValueError(f"{location}: {column} must be finite, got {value!r}")
    if not 0 <= number <= 1:
        raise ValueError(f"{location}: {column} must be in [0, 1], got {value!r}")
    return number


def read_predictions(path, class_count):
    """Yield validated (true ID, predicted ID, confidence, probabilities) records."""
    path = Path(path)
    expected_prob_columns = tuple(f"prob_{i}" for i in range(class_count))
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next((row for row in reader if row), None)
            if header is None:
                raise ValueError(f"{path}: empty CSV, expected a header and prediction rows")
            names = set(header)
            if len(names) != len(header) or any(not name.strip() for name in header):
                raise ValueError(f"{path}: CSV header contains duplicate or blank column names")
            missing = [name for name in REQUIRED_COLUMNS if name not in names]
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

            present_probs = {name for name in names if name.startswith("prob_")}
            unexpected = sorted(present_probs.difference(expected_prob_columns))
            if unexpected:
                raise ValueError(f"{path}: unknown probability columns: {', '.join(unexpected)}")
            prob_columns = ()
            if present_probs:
                missing = [name for name in expected_prob_columns if name not in present_probs]
                if missing:
                    raise ValueError(f"{path}: incomplete probability columns, missing: {', '.join(missing)}")
                prob_columns = expected_prob_columns

            found_row = False
            for fields in reader:
                if not fields:
                    continue
                found_row = True
                location = f"{path}: line {reader.line_num}"
                if len(fields) != len(header):
                    raise ValueError(
                        f"{location}: expected {len(header)} fields, got {len(fields)}"
                    )
                row = dict(zip(header, fields))
                true_id = _class_id(row["true_class_id"], "true_class_id", location, class_count)
                pred_id = _class_id(row["pred_class_id"], "pred_class_id", location, class_count)
                confidence = None
                if "confidence" in names:
                    confidence = _probability(row["confidence"], "confidence", location)
                probabilities = None
                if prob_columns:
                    probabilities = tuple(_probability(row[name], name, location) for name in prob_columns)
                    with localcontext(PROBABILITY_SUM_CONTEXT):
                        total = sum(probabilities)
                        if abs(total - 1) > PROBABILITY_SUM_TOLERANCE:
                            raise ValueError(f"{location}: probabilities must sum to 1 "
                                             f"(within {PROBABILITY_SUM_TOLERANCE:g}), got {total:.12g}")
                    if probabilities[pred_id] != max(probabilities):
                        raise ValueError(f"{location}: pred_class_id {pred_id} does not have "
                                         "a maximum class probability")
                yield true_id, pred_id, confidence, probabilities

            if not found_row:
                raise ValueError(f"{path}: CSV contains no prediction rows")
        except csv.Error as exc:
            raise ValueError(f"{path}: malformed CSV near line {reader.line_num}: {exc}") from exc
