"""Minimal integrity checks for the frozen WildQuest dataset."""
import hashlib
import sys
from unittest import mock
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import get_dataset as ds


class DatasetTests(unittest.TestCase):
    def test_frozen_manifest_is_valid(self):
        rows = ds.load_manifest()
        self.assertEqual(len(rows), 25_000)

    def test_source_priority(self):
        self.assertEqual(
            ds.resolve_source("s3://override/data", {ds.ENV_NAME: "s3://env/data"}, "s3://default/data"),
            "s3://override/data",
        )
        self.assertEqual(
            ds.resolve_source(None, {ds.ENV_NAME: "s3://env/data"}, "s3://default/data"),
            "s3://env/data",
        )
        self.assertEqual(
            ds.resolve_source(None, {}, "s3://default/data"), "s3://default/data",
        )

    def test_s3_prefix_with_at_sign(self):
        uri = "s3://dats-dl/syedibrahim.hamza@gwu.edu/WildQuest/dataset"
        self.assertEqual(ds.resolve_source(uri, {}, ""), uri)

    def test_bad_source_rejected(self):
        with self.assertRaises(ValueError):
            ds.resolve_source(None, {}, "")
        with self.assertRaises(ValueError):
            ds.resolve_source("file:///tmp/data", {}, "")

    def test_single_file_repair_path(self):
        """Missing image is fetched from its manifest-derived S3 object key."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = Path("data/raw/images/train/00/123.jpg")
            image = b"image-bytes"
            rows = [(relative, hashlib.sha256(image).hexdigest())]
            calls = []

            def fake_aws(*arguments):
                calls.append(arguments)
                self.assertEqual(arguments[0:3], ("s3", "cp", "s3://bucket/dataset/images/train/00/123.jpg"))
                Path(arguments[3]).write_bytes(image)

            with (mock.patch.object(ds, "ROOT", root),
                  mock.patch.object(ds, "load_manifest", return_value=rows),
                  mock.patch.object(ds, "aws", side_effect=fake_aws),
                  mock.patch.object(ds.shutil, "which", return_value="/usr/bin/aws"),
                  mock.patch.object(sys, "argv", ["get_dataset.py", "--source", "s3://bucket/dataset"])):
                ds.main()
            self.assertEqual(len(calls), 1)
            self.assertEqual((root / relative).read_bytes(), image)

    def test_valid_dataset_skips_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rel = Path("data/raw/images/train/00/123.jpg")
            file = root / rel
            file.parent.mkdir(parents=True)
            file.write_bytes(b"valid")
            rows = [(rel, hashlib.sha256(b"valid").hexdigest())]
            with (mock.patch.object(ds, "ROOT", root),
                  mock.patch.object(ds, "load_manifest", return_value=rows),
                  mock.patch.object(ds, "aws", side_effect=AssertionError("Unexpected network call")),
                  mock.patch.object(sys, "argv", ["get_dataset.py"])):
                ds.main()

    def test_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rel = Path("data/raw/images/train/00/123.jpg")
            path = root / rel
            digest = hashlib.sha256(b"healthy").hexdigest()
            self.assertEqual(ds.invalid_files([(rel, digest)], root), [rel])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"broken")
            self.assertEqual(ds.invalid_files([(rel, digest)], root), [rel])
            path.write_bytes(b"healthy")
            self.assertEqual(ds.invalid_files([(rel, digest)], root), [])


if __name__ == "__main__":
    unittest.main()
