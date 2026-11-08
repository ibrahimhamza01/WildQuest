#!/usr/bin/env python3
"""Download and verify the fixed WildQuest training dataset from an S3 mirror."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "Code" / "configs"
MANIFEST = CONFIG / "dataset_manifest.csv"
MANIFEST_SHA = CONFIG / "dataset_manifest.sha256"
CLASS_MAPPING = CONFIG / "class_mapping.json"

# Set this ONCE to your team's published S3 prefix before merging.
# An S3 URI identifies the dataset mirror; it is not an AWS credential.
DEFAULT_DATASET_URI = "s3://dats-dl/syedibrahim.hamza@gwu.edu/WildQuest/dataset"
ENV_NAME = "WILDQUEST_DATASET_URI"

SPLIT_COUNTS = {"train": 750, "val": 100, "test": 100, "dmv_test": 50}
CLASS_COUNT = 25
REQUIRED_COLUMNS = {
    "class_id", "split", "photo_id", "observation_uuid", "scientific_name",
    "license", "is_dmv_bbox", "local_path", "file_sha256",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest():
    """Validate the frozen manifest and return (relative image path, sha256) pairs."""
    if not all(p.is_file() for p in (MANIFEST, MANIFEST_SHA, CLASS_MAPPING)):
        raise ValueError("Missing dataset manifest, checksum, or class mapping")

    recorded_sha = MANIFEST_SHA.read_text(encoding="utf-8").split()[0].lower()
    if sha256_file(MANIFEST) != recorded_sha:
        raise ValueError("Manifest SHA-256 mismatch. Do not edit the frozen manifest")

    mapping = json.loads(CLASS_MAPPING.read_text(encoding="utf-8"))
    labels = mapping["id_to_label"]
    if set(labels) != {str(n) for n in range(CLASS_COUNT)}:
        raise ValueError("Class mapping must contain exactly 25 IDs (0-24)")

    rows = []
    counts = Counter()
    seen = {name: set() for name in ("photo_id", "observation_uuid", "file_sha256", "local_path")}
    with MANIFEST.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Missing required manifest columns")
        for line, row in enumerate(reader, start=2):
            class_id = int(row["class_id"])
            split = row["split"]
            if class_id not in range(CLASS_COUNT) or split not in SPLIT_COUNTS:
                raise ValueError(f"Unknown class/split at line {line}")
            if row["scientific_name"] != labels[str(class_id)]:
                raise ValueError(f"Class label mismatch at line {line}")
            if row["license"] not in {"CC0", "CC-BY"}:
                raise ValueError(f"Unsupported photo license at line {line}")
            flag = row["is_dmv_bbox"].strip().lower()
            if flag not in {"true", "false", "0", "1"}:
                raise ValueError(f"Invalid DMV flag at line {line}")
            is_dmv = flag in {"true", "1"}
            if split == "dmv_test" and not is_dmv:
                raise ValueError(f"Non-DMV image in DMV test at line {line}")
            if split == "test" and is_dmv:
                raise ValueError(f"DMV image in general test at line {line}")

            for column, values in seen.items():
                value = row[column]
                if not value or value in values:
                    raise ValueError(f"Missing or repeated {column} at line {line}")
                values.add(value)
            expected_sha = row["file_sha256"].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"Invalid image SHA-256 at line {line}")

            relative = Path(row["local_path"])
            expected_parent = Path("data/raw/images") / split / f"{class_id:02d}"
            if (relative.is_absolute() or ".." in relative.parts or
                    relative.parent != expected_parent or
                    not relative.name.startswith(row["photo_id"] + ".")):
                raise ValueError(f"Invalid local_path at line {line}: {relative}")
            rows.append((relative, expected_sha))
            counts[(class_id, split)] += 1

    if len(rows) != 25_000:
        raise ValueError(f"Expected 25,000 rows, got {len(rows)}")
    for class_id in range(CLASS_COUNT):
        for split, count in SPLIT_COUNTS.items():
            if counts[class_id, split] != count:
                raise ValueError(f"Wrong split size for class {class_id}, {split}")
    return rows


def resolve_source(cli_source=None, env=None, default=DEFAULT_DATASET_URI):
    """Source priority: CLI argument, environment variable, repo default."""
    env = os.environ if env is None else env
    source = (cli_source or env.get(ENV_NAME) or default or "").rstrip("/")
    if not re.fullmatch(r"s3://[a-zA-Z0-9][a-zA-Z0-9.-]{2,62}(?:/[^\s]+)?", source):
        raise ValueError(
            "Set DEFAULT_DATASET_URI in get_dataset.py, set WILDQUEST_DATASET_URI, "
            "or pass --source s3://bucket/prefix"
        )
    return source


def invalid_files(rows, root=None):
    """Return every missing or SHA-256-mismatched relative image path."""
    root = ROOT if root is None else root
    bad = []
    for i, (relative, expected) in enumerate(rows, 1):
        path = root / relative
        try:
            correct = path.is_file() and sha256_file(path) == expected
        except OSError:
            correct = False
        if not correct:
            bad.append(relative)
        if i % 5000 == 0:
            print(f"Checked {i:,}/{len(rows):,} images; missing/corrupt={len(bad)}", flush=True)
    return bad


def aws(*args):
    subprocess.run(["aws", *args], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download/verify the frozen WildQuest dataset")
    parser.add_argument("--source", help="Override the team's S3 dataset prefix")
    parser.add_argument("--verify-only", action="store_true", help="Check local image SHA-256 hashes; no S3 access")
    args = parser.parse_args()

    rows = load_manifest()
    print(f"Manifest OK: {len(rows):,} photos across {CLASS_COUNT} classes", flush=True)
    missing = invalid_files(rows)
    if not missing:
        print(f"SUCCESS: all {len(rows):,} image hashes verified", flush=True)
        return
    if args.verify_only:
        raise RuntimeError(f"Dataset incomplete: {len(missing):,} missing/corrupt images")

    source = resolve_source(args.source)
    if not shutil.which("aws"):
        raise RuntimeError("AWS CLI is required (aws --version)")

    if len(missing) > 50:
        # Efficient for new clones: AWS CLI downloads many objects concurrently.
        target = ROOT / "data" / "raw" / "images"
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset from {source}/images/", flush=True)
        aws("s3", "sync", f"{source}/images/", str(target), "--only-show-errors")
    else:
        # Small repair without syncing or touching the rest of the dataset.
        for relative in missing:
            path = ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            key = relative.relative_to("data/raw/images").as_posix()
            aws("s3", "cp", f"{source}/images/{key}", str(path), "--only-show-errors")

    remaining = invalid_files(rows)
    if remaining and len(missing) > 50 and len(remaining) <= 50:
        # The bulk sync may skip a corrupt file with unchanged size/timestamp.
        print(f"Retrying {len(remaining)} invalid images individually...", flush=True)
        for relative in remaining:
            target = ROOT / relative
            key = relative.relative_to("data/raw/images").as_posix()
            aws("s3", "cp", f"{source}/images/{key}", str(target), "--only-show-errors")
        remaining = invalid_files(rows)
    if remaining:
        for rel in remaining[:5]:
            print(f"INVALID: {rel}", file=sys.stderr)
        raise RuntimeError(f"Download incomplete: {len(remaining):,} images missing/corrupt")
    print(f"SUCCESS: all {len(rows):,} image hashes verified", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
