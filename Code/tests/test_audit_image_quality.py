"""Fast synthetic tests for stratified audit sampling (no actual dataset needed)."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "audit_image_quality.py"
spec = importlib.util.spec_from_file_location("audit_image_quality", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cfg = self.root / "Code/configs"
        cfg.mkdir(parents=True)
        mapping = {str(i): f"Species {i:02d}" for i in range(25)}
        (cfg / "class_mapping.json").write_text(json.dumps({"id_to_label": mapping}))
        manifest = cfg / "dataset_manifest.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "class_id", "scientific_name", "local_path", "observation_uuid"])
            writer.writeheader()
            for split in ("train", "val", "test", "dmv_test"):
                for cls in range(25):
                    for ix in range(3):
                        rel = f"data/raw/images/{split}/{cls:02d}/{ix}.jpg"
                        writer.writerow({"split": split, "class_id": cls, "scientific_name": mapping[str(cls)],
                                         "local_path": rel, "observation_uuid": f"{split}-{cls}-{ix}"})
                        if split in ("train", "val"):
                            img = self.root / rel
                            img.parent.mkdir(parents=True, exist_ok=True)
                            Image.new("RGB", (8, 8), (cls, ix, 200)).save(img)
        (cfg / "dataset_manifest.sha256").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")

    def test_sample_only_allowed_splits_and_deterministic(self):
        records = audit.manifest_records(self.root)
        self.assertEqual(len(records), 50)
        a = audit.choose_samples(records, per_class=2, seed=42)
        b = audit.choose_samples(records, per_class=2, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        self.assertEqual(CounterClassPairs(a), {(s, c): 2 for s in ("train", "val") for c in range(25)})
        self.assertEqual({r["split"] for r in a}, {"train", "val"})

    def test_create_and_summarize_and_no_overwrite(self):
        out = self.root / "outputs/audits/image_quality_v1"
        audit.sample(self.root, out, per_class=1, seed=7)
        self.assertEqual(len(list((out / "sheets").glob("*.png"))), 2)
        with self.assertRaises(FileExistsError):
            audit.sample(self.root, out, per_class=1, seed=7)
        with (out / "review.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 50)
        for row in rows:
            row.update(image_type="visible_animal", visibility="clear", animal_count="single", label_assessment="plausible")
        with (out / "review.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=audit.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        audit.summarize(out)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["splits"]["train"]["fully_reviewed"], 25)
        self.assertEqual(summary["splits"]["val"]["visible_animal_fraction"], 1.0)


def CounterClassPairs(rows):
    from collections import Counter
    return dict(Counter((r["split"], r["class_id"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
