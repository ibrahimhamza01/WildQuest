"""Tests for Code/scripts/audit_dataset.py (issue #4)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_dataset as audit  # noqa: E402


def make_manifest():
    rows = []
    classes = (("Mallard", "Bird"), ("Raccoon", "Mammal"))
    splits = ("train", "train", "train", "val", "test", "dmv_test")
    for cid, (name, group) in enumerate(classes):
        for i, split in enumerate(splits):
            pid = cid * 100 + i
            rows.append(dict(
                class_id=cid, split=split,
                photo_id=pid, observation_uuid=f"obs-{pid}", common_name=name,
                scientific_name=f"Species {cid}", group=group, license="CC0",
                is_dmv_bbox=split == "dmv_test", file_sha256=f"hash-{pid}",
                local_path=f"data/raw/images/{split}/{cid:02d}/{pid}.jpeg",
                downloaded_width=400, downloaded_height=300))
    return pd.DataFrame(rows)


class TestManifestChecks(unittest.TestCase):
    def test_class_distribution_counts_each_split(self):
        dist = audit.class_distribution(make_manifest())
        self.assertEqual(list(dist["total"]), [6, 6])
        self.assertEqual(int(dist["train"].sum()), 6)
        self.assertEqual(int(dist["val"].sum()), 2)
        self.assertEqual(int(dist["test"].sum()), 2)
        self.assertEqual(int(dist["dmv_test"].sum()), 2)

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

    def test_aspect_threshold_uses_unrounded_ratio(self):
        m = make_manifest()
        m.loc[0, ["downloaded_width", "downloaded_height"]] = [1000, 500]
        m.loc[1, ["downloaded_width", "downloaded_height"]] = [500, 1001]
        res = audit.resolution_summary(m)
        self.assertFalse(res.loc[0, "extreme_aspect"])
        self.assertTrue(res.loc[1, "extreme_aspect"])


class TestImageChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.m = make_manifest()
        self.train = self.m.loc[self.m["split"].eq("train")]
        for row in self.m.itertuples():
            path = self.root / row.local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (400, 300), (120, 140, 90)).save(path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_image_audit(self):
        manifest = self.root / "manifest.csv"
        self.m.to_csv(manifest, index=False)
        cmap = self.root / "class_mapping.json"
        cmap.write_text(json.dumps({"label_to_id": {"Species 0": 0, "Species 1": 1}}))
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        audit.main(["--manifest", str(manifest), "--class-map", str(cmap),
                    "--out", str(self.out), "--root", str(self.root), "--images"])
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), before)
        return (self.out / "summary.txt").read_text()

    def test_full_run_opens_only_training_images_and_keeps_all_manifest_splits(self):
        with patch.object(Image, "open", wraps=Image.open) as open_image:
            summary = self.run_image_audit()
        image_root = self.root / "data/raw/images"
        opened = {Path(call.args[0]) for call in open_image.call_args_list
                  if isinstance(call.args[0], (str, Path))
                  and Path(call.args[0]).is_relative_to(image_root)}
        self.assertEqual(opened, {self.root / path for path in self.train["local_path"]})
        dist = pd.read_csv(self.out / "class_distribution.csv")
        self.assertEqual(dist[["train", "val", "test", "dmv_test"]].sum().to_dict(),
                         {"train": 6, "val": 2, "test": 2, "dmv_test": 2})
        info = pd.read_csv(self.out / "image_inspection.csv")
        self.assertEqual(set(info["local_path"]), set(self.train["local_path"]))
        for name in ["summary.txt", "class_distribution.csv", "class_distribution.png",
                     "image_inspection.csv", "species_grid.png"]:
            self.assertTrue((self.out / name).exists(), name)
        self.assertEqual(len(list((self.out / "augmentations").glob("*.png"))), 2)
        self.assertIn("disagrees with class_mapping.json: 0", summary)
        self.assertIn("Training images inspected: 6 of 6", summary)
        self.assertIn("Species grid saved with 2 species", summary)

    def test_training_paths_cannot_point_to_a_held_out_file(self):
        actual_train = self.root / self.m.loc[0, "local_path"]
        held_out = self.m.loc[self.m["split"].eq("dmv_test"), "local_path"].iloc[0]
        mislabeled = self.train.copy()
        mislabeled.loc[0, "local_path"] = held_out
        with self.assertRaisesRegex(ValueError, "leaves the training split"):
            audit.downloaded_training_rows(mislabeled, self.root)

        actual_train.unlink()
        actual_train.symlink_to(self.root / held_out)
        with self.assertRaisesRegex(ValueError, "leaves the training split"):
            audit.downloaded_training_rows(self.train, self.root)

    def test_corrupt_matching_mismatched_and_unknown_dimensions(self):
        (self.root / self.m.loc[0, "local_path"]).write_bytes(b"not an image")
        Image.new("L", (380, 300)).save(self.root / self.m.loc[2, "local_path"])
        self.m.loc[6, "downloaded_width"] = float("nan")

        summary = self.run_image_audit()
        self.assertIn("Training images inspected: 6 of 6", summary)
        self.assertIn("Unreadable images: 1", summary)
        self.assertIn("Size differs from manifest: 1", summary)
        self.assertIn("Size unavailable in manifest: 1", summary)
        self.assertIn("Color modes: RGB=4, L=1", summary)
        self.assertTrue((self.out / "species_grid.png").is_file())

    def test_all_training_images_corrupt_still_write_summary(self):
        for path in self.train["local_path"]:
            (self.root / path).write_bytes(b"not an image")

        summary = self.run_image_audit()
        self.assertIn("Training images inspected: 6 of 6", summary)
        self.assertIn("Unreadable images: 6", summary)
        self.assertIn("Size differs from manifest: 0", summary)
        self.assertIn("Figures skipped: no readable training images.", summary)
        self.assertFalse((self.out / "species_grid.png").exists())
        self.assertFalse((self.out / "augmentations").exists())

    def test_missing_training_images_are_counted_separately(self):
        missing = self.root / self.m.loc[0, "local_path"]
        missing.unlink()
        missing.mkdir()
        (self.root / self.m.loc[1, "local_path"]).unlink()

        summary = self.run_image_audit()
        self.assertIn("Eligible training images found locally: 4 of 6", summary)
        self.assertIn("Missing training images: 2", summary)
        self.assertIn("Training images inspected: 4 of 6", summary)
        self.assertIn("Unreadable images: 0", summary)

    def test_no_local_training_images_still_write_summary(self):
        for path in self.train["local_path"]:
            (self.root / path).unlink()

        summary = self.run_image_audit()
        self.assertIn("Missing training images: 6", summary)
        self.assertIn("Training images inspected: 0 of 6", summary)
        self.assertIn("Unreadable images: 0", summary)
        self.assertIn("Figures skipped: no readable training images.", summary)
        self.assertTrue(pd.read_csv(self.out / "image_inspection.csv").empty)


class TestAugmentationPreviews(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (500, 300), "navy")
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((0, 10, 180, 250), fill="red")
        draw.ellipse((210, 170, 435, 260), fill="orange")

    def test_preview_dimensions_evaluation_and_forced_flips(self):
        panels = [image for _, image in audit._augmentation_panels(self.image)]
        original, evaluation, _, horizontal, _, _, vertical, _ = panels
        self.assertEqual(original.size, (500, 300))
        self.assertTrue(all(image.size == (224, 224) for image in panels[1:]))

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        normalized = transforms.Normalize(weights.mean, weights.std)(transforms.ToTensor()(evaluation))
        torch.testing.assert_close(normalized, weights(self.image))
        self.assertEqual(horizontal.tobytes(), ImageOps.mirror(evaluation).tobytes())
        self.assertEqual(vertical.tobytes(), ImageOps.flip(evaluation).tobytes())

    def test_seeded_previews_match_torchvision_without_changing_rng_state(self):
        rng_state = torch.random.get_rng_state()
        panels = [image for _, image in audit._augmentation_panels(self.image, seed=42)]
        same_seed = audit._augmentation_panels(self.image, seed=42)
        different_seed = audit._augmentation_panels(self.image, seed=43)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertEqual([image.tobytes() for image in panels],
                         [image.tobytes() for _, image in same_seed])
        self.assertNotEqual(panels[2].tobytes(), different_seed[2][1].tobytes())

        crop = transforms.RandomResizedCrop(224, scale=(0.6, 1.0),
                                            interpolation=transforms.InterpolationMode.BICUBIC)
        jitter = transforms.ColorJitter(0.3, 0.3, 0.3)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.assertEqual(panels[2].tobytes(), crop(self.image).tobytes())
            evaluation = panels[1]
            transforms.RandomHorizontalFlip(p=1.0)(evaluation)
            self.assertEqual(panels[4].tobytes(), jitter(evaluation).tobytes())
            self.assertEqual(panels[5].tobytes(), transforms.RandomRotation(15)(evaluation).tobytes())
            transforms.RandomVerticalFlip(p=1.0)(evaluation)
            training = transforms.Compose([crop, transforms.RandomHorizontalFlip(p=0.5), jitter])
            self.assertEqual(panels[7].tobytes(), training(self.image).tobytes())


if __name__ == "__main__":
    unittest.main()
