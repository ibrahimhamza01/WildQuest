"""Tests for Code/scripts/audit_dataset.py (issue #4)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_dataset as audit  # noqa: E402


def make_manifest(n_per_class=4, classes=(("Mallard", "Bird"), ("Raccoon", "Mammal"))):
    rows = []
    for cid, (name, group) in enumerate(classes):
        for i in range(n_per_class):
            pid = cid * 100 + i
            rows.append(dict(
                class_id=cid, split="train" if i < n_per_class - 1 else "val",
                photo_id=pid, observation_uuid=f"obs-{pid}", common_name=name,
                scientific_name=f"Species {cid}", group=group, license="CC0",
                is_dmv_bbox=False, file_sha256=f"hash-{pid}",
                local_path=f"data/raw/images/{pid}.jpeg",
                downloaded_width=400, downloaded_height=300))
    return pd.DataFrame(rows)


class TestManifestChecks(unittest.TestCase):
    def test_class_distribution_counts_each_split(self):
        dist = audit.class_distribution(make_manifest())
        self.assertEqual(list(dist["total"]), [4, 4])
        self.assertEqual(int(dist["train"].sum()), 6)
        self.assertEqual(int(dist["val"].sum()), 2)

    def test_leakage_check_finds_observation_in_two_splits(self):
        m = make_manifest()
        self.assertEqual(audit.leakage_check(m), [])
        m.loc[1, "observation_uuid"] = m.loc[3, "observation_uuid"]  # train + val
        self.assertEqual(len(audit.leakage_check(m)), 1)

    def test_duplicate_files_detected_by_hash(self):
        m = make_manifest()
        self.assertEqual(len(audit.duplicate_files(m)), 0)
        m.loc[1, "file_sha256"] = m.loc[0, "file_sha256"]
        self.assertEqual(len(audit.duplicate_files(m)), 2)

    def test_resolution_flags_small_and_extreme_aspect(self):
        m = make_manifest()
        m.loc[0, ["downloaded_width", "downloaded_height"]] = [100, 150]
        m.loc[1, ["downloaded_width", "downloaded_height"]] = [900, 300]
        res = audit.resolution_summary(m)
        self.assertTrue(res.loc[0, "too_small"])
        self.assertTrue(res.loc[1, "extreme_aspect"])
        self.assertFalse(res.loc[2, "too_small"] or res.loc[2, "extreme_aspect"])


class TestImageChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.m = make_manifest()
        for row in self.m.itertuples():
            path = self.root / row.local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (400, 300), (120, 140, 90)).save(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inspect_images_reads_files_and_flags_corrupt(self):
        (self.root / self.m.loc[0, "local_path"]).write_bytes(b"not an image")
        info = audit.inspect_images(self.m, self.root)
        self.assertEqual(int((~info["readable"]).sum()), 1)
        ok = info[info["readable"]]
        self.assertTrue(ok["dims_match_manifest"].all())
        self.assertEqual(set(ok["mode"]), {"RGB"})

    def test_downloaded_rows_skips_missing_files(self):
        (self.root / self.m.loc[0, "local_path"]).unlink()
        self.assertEqual(len(audit.downloaded_rows(self.m, self.root)), len(self.m) - 1)

    def test_full_run_writes_outputs_and_leaves_manifest_unchanged(self):
        manifest = self.root / "manifest.csv"
        self.m.to_csv(manifest, index=False)
        cmap = self.root / "class_mapping.json"
        cmap.write_text(json.dumps({"label_to_id": {"Species 0": 0, "Species 1": 1}}))
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        out = self.root / "out"

        audit.main(["--manifest", str(manifest), "--class-map", str(cmap),
                    "--out", str(out), "--root", str(self.root), "--images"])

        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), before)
        for name in ["summary.txt", "class_distribution.csv", "class_distribution.png",
                     "image_inspection.csv", "species_grid.png"]:
            self.assertTrue((out / name).exists(), name)
        self.assertEqual(len(list((out / "augmentations").glob("*.png"))), 2)
        self.assertIn("disagrees with class_mapping.json: 0",
                      (out / "summary.txt").read_text())


if __name__ == "__main__":
    unittest.main()