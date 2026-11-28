"""Tests of classifier, pipeline, and CLI contracts without external models."""

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from torch import nn

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from wildquest.classifier.dataset import image_transforms
from wildquest.detection import Detection, MegaDetector
from wildquest.inference import (
    ClassificationResult, EfficientNetClassifier, InferencePipeline, SpeciesPrediction,
    classifier as adapter,
)

ADAPTER = "wildquest.inference.classifier"
CLI = CODE / "scripts/predict_image.py"
cli_spec = importlib.util.spec_from_file_location("predict_image", CLI)
predict_cli = importlib.util.module_from_spec(cli_spec)
cli_spec.loader.exec_module(predict_cli)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(25, dtype=torch.float64))

    def forward(self, images):
        self.input = images
        self.inference_mode = torch.is_inference_mode_enabled()
        return self.logits.unsqueeze(0).expand(images.shape[0], -1)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "best.pt"

        self.logits = torch.linspace(-4, -2, 25, dtype=torch.float64)
        self.logits[17], self.logits[0], self.logits[24] = 4.75, 3.5, 2.25
        self.metadata = {
            "class_mapping": json.loads((CODE / "configs/class_mapping.json").read_text()),
            "manifest_sha256": (CODE / "configs/dataset_manifest.sha256").read_text().split()[0],
            "model_state_dict": {"logits": self.logits},
            "epoch": 1,
            "config": {"seed": 42},
        }
        self.model = TinyClassifier()
        build_model = patch(f"{ADAPTER}.build_model", return_value=self.model)
        self.build_model = build_model.start()
        self.addCleanup(build_model.stop)

    def test_cpu_prediction_uses_checkpoint_once_and_returns_ranked_species(self):
        torch.save(self.metadata, self.path)
        crop = Image.new("RGB", (40, 25), color=(30, 90, 120))
        with patch(f"{ADAPTER}.torch.load", wraps=torch.load) as load:
            classifier = EfficientNetClassifier(self.path, device="cpu")
            result = classifier.predict(crop)
            self.assertEqual(classifier.predict(crop), result)

        self.build_model.assert_called_once_with(num_classes=25, pretrained=False)
        load.assert_called_once_with(self.path, map_location="cpu", weights_only=True)
        self.assertFalse(self.model.training)
        self.assertTrue(self.model.inference_mode)
        self.assertEqual(self.model.input.device, torch.device("cpu"))
        self.assertEqual(result.primary, result.top3[0])

        expected_classes = [(17, "Procyon lotor"), (0, "Anas platyrhynchos"), (24, "Zenaida macroura")]
        actual_classes = [(item.class_id, item.scientific_name) for item in result.top3]
        self.assertEqual(actual_classes, expected_classes)
        probabilities = self.logits.softmax(dim=0)
        for prediction in result.top3:
            self.assertIs(type(prediction.confidence), float)
            self.assertEqual(prediction.confidence, probabilities[prediction.class_id].item())
        payload = json.loads(json.dumps(asdict(result)))
        self.assertEqual(payload["primary"], payload["top3"][0])

    def test_prediction_uses_the_validation_transform(self):
        torch.save(self.metadata, self.path)
        crop = Image.linear_gradient("L").resize((350, 247)).convert("RGB")
        with patch(f"{ADAPTER}.image_transforms", wraps=image_transforms) as transforms:
            classifier = EfficientNetClassifier(self.path)
            classifier.predict(crop)

        transforms.assert_called_once_with("val")
        torch.testing.assert_close(
            self.model.input, image_transforms("val")(crop).unsqueeze(0), rtol=0, atol=0
        )

    def test_prediction_converts_non_rgb_crops(self):
        torch.save(self.metadata, self.path)
        classifier = EfficientNetClassifier(self.path)
        crop = Image.linear_gradient("L")
        classifier.predict(crop)

        self.assertEqual(crop.mode, "L")
        torch.testing.assert_close(
            self.model.input, image_transforms("val")(crop.convert("RGB")).unsqueeze(0), rtol=0, atol=0
        )

    def test_rejects_invalid_crops(self):
        torch.save(self.metadata, self.path)
        classifier = EfficientNetClassifier(self.path)
        with self.assertRaisesRegex(TypeError, "PIL image"):
            classifier.predict("crop.jpg")
        with self.assertRaisesRegex(ValueError, "nonzero"):
            classifier.predict(Image.new("RGB", (0, 32)))

    def test_rejects_missing_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint not found.*best.pt"):
            EfficientNetClassifier(self.path)
        self.build_model.assert_not_called()

    def test_rejects_corrupt_checkpoint(self):
        self.path.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(ValueError, "Cannot load classifier checkpoint.*best.pt"):
            EfficientNetClassifier(self.path)

    def test_rejects_malformed_checkpoint_metadata(self):
        cases = [([], "metadata dictionary"), ({}, "missing class_mapping")]
        for field in ("manifest_sha256", "model_state_dict"):
            cases.append(({key: value for key, value in self.metadata.items() if key != field},
                          f"missing {field}"))
        for state_dict in ([], {0: self.logits}):
            cases.append(({**self.metadata, "model_state_dict": state_dict}, "model state dictionary"))
        for checkpoint, message in cases:
            with self.subTest(message=message):
                torch.save(checkpoint, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    EfficientNetClassifier(self.path)

    def test_rejects_valid_mapping_with_swapped_species(self):
        mapping = self.metadata["class_mapping"]
        labels = mapping["id_to_label"]
        labels["0"], labels["1"] = labels["1"], labels["0"]
        mapping["label_to_id"] = {name: int(class_id) for class_id, name in labels.items()}
        torch.save(self.metadata, self.path)

        with self.assertRaisesRegex(ValueError, "class mapping differs from repository"):
            EfficientNetClassifier(self.path)

    def test_rejects_incomplete_ids_and_invalid_reverse_mapping(self):
        labels = self.metadata["class_mapping"]["id_to_label"]
        reverse = self.metadata["class_mapping"]["label_to_id"]
        incomplete = {class_id: name for class_id, name in labels.items() if class_id != "0"}
        for id_to_label, label_to_id, error in (
            (incomplete, reverse, "contiguous class IDs"),
            (labels, {**reverse, labels["0"]: 1}, "back to their class IDs"),
            (labels, {**reverse, labels["0"]: False}, "back to their class IDs"),
        ):
            with self.subTest(error=error):
                self.metadata["class_mapping"] = {
                    "id_to_label": id_to_label, "label_to_id": label_to_id,
                }
                torch.save(self.metadata, self.path)
                with self.assertRaisesRegex(ValueError, error):
                    EfficientNetClassifier(self.path)

    def test_rejects_manifest_hash_mismatch(self):
        self.metadata["manifest_sha256"] = "0" * 64
        torch.save(self.metadata, self.path)

        with self.assertRaisesRegex(ValueError, "manifest hash differs from frozen local manifest"):
            EfficientNetClassifier(self.path)

    def test_strictly_rejects_state_dict_mismatches(self):
        for state_dict, detail in (
            ({}, "Missing key"),
            ({"logits": self.logits, "extra": torch.zeros(1)}, "Unexpected key"),
            ({"logits": torch.zeros(24)}, "size mismatch"),
        ):
            with self.subTest(detail=detail):
                torch.save({**self.metadata, "model_state_dict": state_dict}, self.path)
                with self.assertRaisesRegex(ValueError, f"(?s)Incompatible EfficientNet-B0.*{detail}"):
                    EfficientNetClassifier(self.path)

    def test_rejects_unsupported_device(self):
        with self.assertRaisesRegex(ValueError, "Unsupported classifier device meta"):
            EfficientNetClassifier(self.path, device="meta")

    def test_rejects_unavailable_cuda(self):
        with patch(f"{ADAPTER}.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA device cuda is unavailable"):
                EfficientNetClassifier(self.path, device="cuda")

    def test_rejects_cuda_index_outside_device_count(self):
        with (
            patch(f"{ADAPTER}.torch.cuda.is_available", return_value=True),
            patch(f"{ADAPTER}.torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA device cuda:1.*index is out of range"):
                EfficientNetClassifier(self.path, device="cuda:1")

    def test_resolves_unindexed_cuda_to_current_gpu(self):
        with (
            patch(f"{ADAPTER}.torch.cuda.is_available", return_value=True),
            patch(f"{ADAPTER}.torch.cuda.current_device", return_value=1),
        ):
            self.assertEqual(adapter._resolve_device("cuda"), torch.device("cuda:1"))

    def test_oversized_cuda_index_does_not_wrap_to_another_device(self):
        for device in ("cuda:128", "cuda:255", "cuda:256"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "index.*out of range"):
                EfficientNetClassifier(self.path, device=device)

    def test_rejects_invalid_model_outputs(self):
        torch.save(self.metadata, self.path)
        classifier = EfficientNetClassifier(self.path)
        crop = Image.new("RGB", (32, 32))
        for logits, error in (
            (torch.zeros((25,)), "logits of shape"),
            (torch.zeros((1, 24)), "logits of shape"),
            (torch.zeros((1, 25), dtype=torch.long), "floating-point logits"),
            (torch.full((1, 25), float("nan")), "non-finite logits"),
            (torch.full((1, 25), float("inf")), "non-finite logits"),
        ):
            with self.subTest(error=error), patch.object(self.model, "forward", return_value=logits):
                with self.assertRaisesRegex(RuntimeError, error):
                    classifier.predict(crop)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (6, 4))
        self.image.putdata([(x, y, 20) for y in range(4) for x in range(6)])
        self.detector = Mock(spec=MegaDetector)
        self.classifier = Mock(spec=EfficientNetClassifier)
        self.pipeline = InferencePipeline(self.detector, self.classifier)

        goose = SpeciesPrediction(2, "Branta canadensis", 0.73456123456)
        deer = SpeciesPrediction(13, "Odocoileus virginianus", 0.2)
        robin = SpeciesPrediction(22, "Turdus migratorius", 0.06)
        self.goose = ClassificationResult(goose, (goose, deer, robin))
        deer = SpeciesPrediction(13, "Odocoileus virginianus", 0.9)
        goose = SpeciesPrediction(2, "Branta canadensis", 0.03)
        self.deer = ClassificationResult(deer, (deer, robin, goose))
        self.first = Detection("animal", 0.876543219, (1, 0, 4, 3))
        self.second = Detection("animal", 0.95, (2, 1, 6, 4))
        self.detector.detect.return_value = [self.first]
        self.classifier.predict.return_value = self.goose

    def test_single_detection_crops_the_expected_pixels_and_serializes(self):
        result = self.pipeline.predict(self.image)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.oriented_size, (6, 4))
        self.assertEqual(result.errors, ())
        self.assertEqual(len(result.predictions), 1)
        self.assertIs(result.predictions[0].detection, self.first)
        self.assertIs(result.predictions[0].classification, self.goose)
        self.detector.detect.assert_called_once()
        self.classifier.predict.assert_called_once()
        crop = self.classifier.predict.call_args.args[0]
        self.assertEqual(crop.size, (3, 3))
        self.assertEqual(list(crop.get_flattened_data()),
                         [(x, y, 20) for y in range(3) for x in range(1, 4)])

        payload = result.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        self.assertEqual(payload["oriented_size"], [6, 4])
        self.assertEqual(payload["predictions"], [{
            "box_xyxy": [1, 0, 4, 3],
            "category": "animal",
            "detector_confidence": 0.876543219,
            "class_id": 2,
            "scientific_name": "Branta canadensis",
            "classification_confidence": 0.73456123456,
            "top3": [asdict(item) for item in self.goose.top3],
        }])

    def test_multiple_overlapping_detections_preserve_associations_and_reuse_models(self):
        self.detector.detect.return_value = [self.first, self.second]
        self.classifier.predict.side_effect = [self.goose, self.deer, self.goose, self.deer]

        result = self.pipeline.predict(self.image)
        self.assertEqual(self.pipeline.predict(self.image), result)

        self.assertEqual(result.status, "ok")
        self.assertEqual([(item.detection, item.classification) for item in result.predictions],
                         [(self.first, self.goose), (self.second, self.deer)])
        self.assertEqual(self.detector.detect.call_count, 2)
        self.assertEqual(self.classifier.predict.call_count, 4)
        crops = [call.args[0] for call in self.classifier.predict.call_args_list]
        self.assertEqual([crop.size for crop in crops], [(3, 3), (4, 3)] * 2)
        self.assertEqual([crop.getpixel((0, 0)) for crop in crops], [(1, 0, 20), (2, 1, 20)] * 2)

    def test_no_detection_leaves_predictions_empty(self):
        self.detector.detect.return_value = []

        result = self.pipeline.predict(self.image)

        self.assertEqual(result.to_dict(), {
            "status": "no_detection", "oriented_size": [6, 4], "predictions": [], "errors": [],
        })
        self.classifier.predict.assert_not_called()

    def test_megadetector_filters_bad_boxes_before_pipeline_status_and_indexing(self):
        malformed = SimpleNamespace(label="animal", confidence=0.9, bbox=SimpleNamespace(
            x_min=0.5, y_min=0, x_max=0, y_max=1,
        ))
        left = SimpleNamespace(label="animal", confidence=0.9, bbox=SimpleNamespace(
            x_min=0, y_min=0, x_max=0.5, y_max=1,
        ))
        right = SimpleNamespace(label="animal", confidence=0.9, bbox=SimpleNamespace(
            x_min=0.5, y_min=0, x_max=1, y_max=1,
        ))
        pipeline = InferencePipeline(MegaDetector(), self.classifier)
        with patch("wildquest.detection.megadetector._sparrow_predictions") as backend:
            backend.return_value = [malformed]
            empty = pipeline.predict(self.image)
            self.assertEqual((empty.status, empty.predictions, empty.errors), ("no_detection", (), ()))
            self.classifier.predict.assert_not_called()

            backend.return_value = [malformed, left, right]
            self.classifier.predict.side_effect = [self.goose, RuntimeError("classification failed")]
            partial = pipeline.predict(self.image)

        self.assertEqual(partial.status, "partial")
        self.assertEqual([item.detection.box for item in partial.predictions], [(0, 0, 3, 4)])
        self.assertEqual([(error.stage, error.detection_index) for error in partial.errors],
                         [("classifier", 1)])

    def test_path_and_pil_inputs_use_the_same_exif_oriented_rgb_pixels(self):
        original = Image.new("L", (3, 2))
        original.putpixel((0, 0), 220)
        original.getexif()[274] = 6  # Rotate 90 degrees clockwise for display.
        self.detector.detect.return_value = [Detection("animal", 0.8, (1, 0, 2, 1))]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oriented.png"
            original.save(path, exif=original.getexif())
            for image in (path, str(path), original):
                with self.subTest(image=type(image)):
                    result = self.pipeline.predict(image)
                    self.assertEqual(result.status, "ok")
                    self.assertEqual(result.oriented_size, (2, 3))
                    detector_input = self.detector.detect.call_args.args[0]
                    crop = self.classifier.predict.call_args.args[0]
                    self.assertEqual((detector_input.mode, detector_input.size), ("RGB", (2, 3)))
                    self.assertEqual(detector_input.getpixel((1, 0)), (220, 220, 220))
                    self.assertEqual((crop.size, crop.getpixel((0, 0))), ((1, 1), (220, 220, 220)))
        self.assertEqual((original.mode, original.size, original.getexif()[274]), ("L", (3, 2), 6))

    def test_bad_images_return_input_errors_before_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.jpg"
            malformed = Path(temp) / "malformed.jpg"
            malformed.write_bytes(b"invalid image contents")
            for image, message in (
                (missing, "missing.jpg"),
                (malformed, "malformed.jpg"),
                (object(), "filesystem path or PIL image"),
                (Image.new("RGB", (0, 4)), "positive width and height"),
            ):
                with self.subTest(message=message):
                    result = self.pipeline.predict(image)
                    self.assertEqual(result.status, "error")
                    self.assertIsNone(result.oriented_size)
                    self.assertEqual(result.predictions, ())
                    self.assertEqual(result.errors[0].stage, "input")
                    self.assertIn(message, result.errors[0].message)
                    self.assertEqual(result.to_dict()["errors"][0]["detection_index"], None)
        self.detector.detect.assert_not_called()
        self.classifier.predict.assert_not_called()

    def test_invalid_boxes_are_reported_and_valid_detections_are_classified(self):
        self.detector.detect.return_value = [
            Detection("animal", 0.8, box) for box in (
                (2, 1, 2, 3), (-1, 1, 2, 3), (0, 0, 7, 4), (0, 1, 2.5, 3), (False, 0, 3, 3),
            )
        ] + [self.second]

        result = self.pipeline.predict(self.image)

        self.assertEqual(result.status, "partial")
        self.assertEqual(tuple(item.detection for item in result.predictions), (self.second,))
        self.assertEqual([(error.stage, error.detection_index) for error in result.errors],
                         [("crop", index) for index in range(5)])
        self.classifier.predict.assert_called_once()
        self.assertEqual(self.classifier.predict.call_args.args[0].size, (4, 3))

    def test_empty_crop_is_reported_without_calling_classifier(self):
        for crop in (None, Image.new("RGB", (0, 3))):
            with self.subTest(crop=crop), patch("wildquest.inference.pipeline.crop_animal", return_value=crop):
                result = self.pipeline.predict(self.image)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.predictions, ())
                self.assertEqual(result.errors[0].stage, "crop")
                self.assertIn("nonempty crop", result.errors[0].message)
        self.classifier.predict.assert_not_called()

    def test_detector_runtime_failure_is_distinct_from_no_detection(self):
        self.detector.detect.side_effect = RuntimeError("detector backend failed")

        result = self.pipeline.predict(self.image)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.oriented_size, (6, 4))
        self.assertEqual(result.predictions, ())
        self.assertEqual(result.to_dict()["errors"], [{
            "stage": "detector", "error_type": "RuntimeError", "message": "detector backend failed",
            "detection_index": None,
        }])
        self.classifier.predict.assert_not_called()

    def test_classifier_failure_preserves_other_crops_and_reports_the_failed_index(self):
        self.detector.detect.return_value = [self.first, self.second]
        self.classifier.predict.side_effect = [RuntimeError("classifier backend failed"), self.deer]

        result = self.pipeline.predict(self.image)

        self.assertEqual(result.status, "partial")
        self.assertEqual([(item.detection, item.classification) for item in result.predictions],
                         [(self.second, self.deer)])
        payload = result.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        self.assertEqual(payload["errors"], [{
            "stage": "classifier", "error_type": "RuntimeError", "message": "classifier backend failed",
            "detection_index": 0,
        }])

    def test_single_classifier_failure_returns_error(self):
        self.classifier.predict.side_effect = RuntimeError("classifier backend failed")

        result = self.pipeline.predict(self.image)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.predictions, ())
        self.assertEqual(result.errors[0].stage, "classifier")

    def test_invalid_detector_output_is_reported_as_an_error(self):
        for detections in (None, [object()]):
            with self.subTest(detections=detections):
                self.detector.detect.return_value = detections
                result = self.pipeline.predict(self.image)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.errors[0].stage, "detector")
        self.classifier.predict.assert_not_called()


class PredictionCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.photo = self.root / "photo.png"
        Image.new("RGB", (6, 4)).save(self.photo)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"placeholder checkpoint")
        self.bundle = self.root / "models/MDV6-yolov10-e"
        self.bundle.mkdir(parents=True)
        self.onnx = self.bundle / "model.onnx"
        self.onnx.write_bytes(b"placeholder ONNX weights")
        (self.bundle / "manifest.toml").write_text(
            '[model]\nid = "MDV6-yolov10-e"\nformat = "onnx"\nfile = "model.onnx"\n'
        )
        self.engine = SimpleNamespace(init=Mock(), active_device=Mock(return_value="cpu"))
        self.ort = SimpleNamespace(get_available_providers=Mock(return_value=["CUDAExecutionProvider"]))
        modules = patch.dict(sys.modules, {"sparrow_engine": self.engine, "onnxruntime": self.ort})
        modules.start()
        self.addCleanup(modules.stop)
        detector = patch.object(predict_cli, "MegaDetector")
        self.build_detector = detector.start()
        self.addCleanup(detector.stop)
        classifier = patch("wildquest.inference.EfficientNetClassifier")
        self.build_classifier = classifier.start()
        self.addCleanup(classifier.stop)
        self.detector = self.build_detector.return_value
        self.classifier = self.build_classifier.return_value
        self.first = Detection("animal", 0.8, (0, 0, 3, 3))
        self.second = Detection("animal", 0.7, (2, 1, 6, 4))
        goose = SpeciesPrediction(2, "Branta canadensis", 0.75)
        robin = SpeciesPrediction(22, "Turdus migratorius", 0.1)
        deer = SpeciesPrediction(13, "Odocoileus virginianus", 0.05)
        self.classification = ClassificationResult(goose, (goose, robin, deer))
        self.detector.detect.return_value = [self.first]
        self.classifier.predict.return_value = self.classification

    def args(self, *extra, image=None, bundle=None):
        return ["--image", str(image or self.photo), "--checkpoint", str(self.checkpoint),
                "--model-bundle", str(bundle or self.bundle), *extra]

    def run_cli(self, *extra, **paths):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            code = predict_cli.main(self.args(*extra, **paths))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_argument_defaults_and_invalid_values(self):
        parsed = predict_cli.parse_args(["--image", str(self.photo), "--model-bundle", str(self.bundle)])
        self.assertEqual(parsed.checkpoint, predict_cli.ROOT / predict_cli.DEFAULT_CHECKPOINT)
        self.assertEqual((parsed.threshold, parsed.detector_device, parsed.classifier_device), (0.2, "cpu", "cpu"))
        for args in ([], self.args("--threshold", "nan"), self.args("--threshold", "1.1"),
                     self.args("--detector-device", "auto"), self.args("--classifier-device", "cuda:-1")):
            with self.subTest(args=args), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    predict_cli.main(args)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), "")
        self.engine.init.assert_not_called()

    def test_configures_each_requested_device_and_bundle_parent(self):
        for detector, classifier, expected_detector, expected_classifier in (
            ("cpu", "cpu", "cpu", "cpu"),
            ("cpu", "cuda", "cpu", "cuda:0"),
            ("cuda:1", "cuda:2", "cuda:1", "cuda:2"),
        ):
            with self.subTest(detector=detector, classifier=classifier):
                self.engine.active_device.return_value = expected_detector
                code, stdout, _ = self.run_cli(
                    "--detector-device", detector, "--classifier-device", classifier, "--threshold", "0.4"
                )
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout)["status"], "ok")
                self.engine.init.assert_called_with(device=expected_detector, model_dir=str(self.bundle.parent))
                self.build_detector.assert_called_with(confidence_threshold=0.4)
                self.build_classifier.assert_called_with(self.checkpoint, device=expected_classifier)

    def test_single_detection_stdout_contains_only_json(self):
        self.engine.init.side_effect = lambda **kwargs: print("engine diagnostic")

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 0)
        self.assertIn("engine diagnostic", stderr)
        self.assertIn("Detector on cpu, classifier on cpu", stderr)
        payload = json.loads(stdout)
        self.assertEqual(set(payload), {"status", "oriented_size", "predictions", "errors"})
        self.assertEqual(payload["oriented_size"], [6, 4])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["predictions"], [{
            "box_xyxy": [0, 0, 3, 3], "category": "animal", "detector_confidence": 0.8,
            "class_id": 2, "scientific_name": "Branta canadensis", "classification_confidence": 0.75,
            "top3": [asdict(item) for item in self.classification.top3],
        }])

    def test_file_output_contains_both_detections_and_refuses_overwrite(self):
        self.detector.detect.return_value = [self.first, self.second]
        path = self.root / "prediction.json"

        code, stdout, _ = self.run_cli("--output", str(path))

        self.assertEqual((code, stdout), (0, ""))
        original = path.read_bytes()
        self.assertEqual([item["box_xyxy"] for item in json.loads(original)["predictions"]],
                         [list(self.first.box), list(self.second.box)])
        code, stdout, stderr = self.run_cli("--output", str(path))
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("File exists", stderr)
        self.assertEqual(path.read_bytes(), original)

    def test_no_detection_is_successful_with_no_classification(self):
        self.detector.detect.return_value = []
        code, stdout, _ = self.run_cli()

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), {
            "status": "no_detection", "oriented_size": [6, 4], "predictions": [], "errors": [],
        })
        self.classifier.predict.assert_not_called()

    def test_missing_unreadable_and_held_out_photos_report_input_errors(self):
        corrupt = self.root / "corrupt.jpg"
        corrupt.write_bytes(b"not a photo")
        with patch.object(predict_cli, "ROOT", self.root):
            for image in (self.root / "missing.jpg", corrupt, self.root / "data/raw/images/test/example.jpg",
                          self.root / "data/raw/images/dmv_test/example.jpg"):
                with self.subTest(image=image):
                    code, stdout, stderr = self.run_cli(image=image)
                    self.assertEqual(code, 2)
                    self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "input")
                    self.assertIn(image.name, stderr)
        self.detector.detect.assert_not_called()

    def test_missing_or_invalid_artifacts_have_actionable_errors(self):
        self.build_classifier.side_effect = EfficientNetClassifier
        for checkpoint in (str(self.root / "missing.pt"), str(self.checkpoint)):
            with self.subTest(checkpoint=checkpoint):
                code, stdout, stderr = self.run_cli("--checkpoint", checkpoint)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "classifier")
                self.assertIn(Path(checkpoint).name, stderr)

        self.onnx.rename(self.bundle / "moved.onnx")
        for bundle, message in ((self.bundle, "ONNX weights not found"),
                                (self.root / "missing/MDV6-yolov10-e", "manifest.toml")):
            with self.subTest(bundle=bundle):
                code, stdout, stderr = self.run_cli(bundle=bundle)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "detector")
                self.assertIn(message, stderr)

    def test_partial_failure_has_predictions_and_distinct_exit_code(self):
        self.detector.detect.return_value = [self.first, self.second]
        self.classifier.predict.side_effect = [RuntimeError("crop failed"), self.classification]

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 3)
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["predictions"][0]["box_xyxy"], list(self.second.box))
        self.assertEqual(payload["errors"][0]["detection_index"], 0)
        self.assertIn("crop failed", stderr)

    def test_runtime_failures_are_fatal_and_stage_specific(self):
        self.detector.detect.side_effect = RuntimeError("detector inference failed")
        code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "detector")
        self.assertIn("detector inference failed", stderr)

        self.detector.detect.side_effect = None
        self.classifier.predict.side_effect = RuntimeError("classifier inference failed")
        code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "classifier")
        self.assertIn("classifier inference failed", stderr)

        self.build_classifier.side_effect = RuntimeError("CUDA unavailable")
        code, stdout, stderr = self.run_cli("--classifier-device", "cuda")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "classifier")
        self.assertIn("CUDA unavailable", stderr)

    def test_cuda_detector_errors_and_device_fallback_are_reported(self):
        self.ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        code, stdout, stderr = self.run_cli("--detector-device", "cuda")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "detector")
        self.assertIn("CUDAExecutionProvider is unavailable", stderr)
        self.engine.init.assert_not_called()

        self.ort.get_available_providers.return_value = ["CUDAExecutionProvider"]
        self.engine.init.side_effect = RuntimeError("CUDA initialization failed")
        code, stdout, stderr = self.run_cli("--detector-device", "cuda")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["errors"][0]["stage"], "detector")
        self.assertIn("CUDA initialization failed", stderr)

        self.engine.init.side_effect = None
        for devices in (["cpu"], ["cuda:0", "cpu"]):
            with self.subTest(devices=devices):
                self.engine.active_device.side_effect = devices
                code, stdout, stderr = self.run_cli("--detector-device", "cuda")
                self.assertEqual(code, 1)
                payload = json.loads(stdout)
                self.assertEqual(payload["status"], "error")
                self.assertEqual(payload["predictions"], [])
                self.assertIn("Requested detector device cuda:0", stderr)

    def test_executable_redirects_native_runtime_output(self):
        script = """
import os
import runpy
import sys
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, sys.argv[1])
import wildquest.inference as inference
inference.EfficientNetClassifier = lambda *args, **kwargs: object()

def detect(path, **kwargs):
    os.write(1, b'native engine diagnostic\\n')
    with Image.open(path) as image:
        return [SimpleNamespace(image_size=image.size, detections=[])]

sys.modules['sparrow_engine'] = SimpleNamespace(
    init=lambda **kwargs: None, active_device=lambda: 'cpu', detect=detect,
)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(CODE / "src"), str(CLI), *self.args()],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "no_detection")
        self.assertIn("native engine diagnostic", completed.stderr)


if __name__ == "__main__":
    unittest.main()
