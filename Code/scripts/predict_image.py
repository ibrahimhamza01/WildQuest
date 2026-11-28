#!/usr/bin/env python3
"""Detect and classify animals in one wildlife photograph."""

import argparse
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.detection.megadetector import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD, MODEL_ID, MegaDetector, read_model_bundle,
)

DEFAULT_CHECKPOINT = Path("outputs/baselines/efficientnet_b0_v1/best.pt")


def _device(value):
    if value == "cpu":
        return value
    match = re.fullmatch(r"cuda(?::(0|[1-9][0-9]*))?", value)
    if match:
        return f"cuda:{match[1] or '0'}"
    raise argparse.ArgumentTypeError("expected cpu, cuda, or cuda:N with a nonnegative GPU index")


def _threshold(value):
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return threshold


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--image", type=Path, required=True, help="Local wildlife photograph")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT,
                        help=f"Local EfficientNet best.pt (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--model-bundle", type=Path, required=True,
                        help=f"Unpacked local {MODEL_ID} ONNX bundle directory")
    parser.add_argument("--threshold", type=_threshold, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Minimum animal confidence (default: {DEFAULT_CONFIDENCE_THRESHOLD:.2f})")
    parser.add_argument("--detector-device", type=_device, default="cpu", metavar="DEVICE",
                        help="MegaDetector device, cpu or cuda[:N] (default: cpu)")
    parser.add_argument("--classifier-device", type=_device, default="cpu", metavar="DEVICE",
                        help="EfficientNet device, cpu or cuda[:N] (default: cpu)")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file instead of stdout")
    return parser.parse_args(argv)


def _image_path(path):
    requested = Path(os.path.abspath(path))
    resolved = requested.resolve()
    image_dir = ROOT / "data/raw/images"
    for split in ("test", "dmv_test"):
        held_out = image_dir / split
        if requested.is_relative_to(held_out) or resolved.is_relative_to(held_out.resolve()):
            raise ValueError(f"Held-out photographs cannot be used for inference: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Photograph file not found or not a regular file: {path}")
    return resolved


def _check_detector_device(engine, expected):
    actual = engine.active_device()
    if actual != expected:
        raise RuntimeError(
            f"Requested detector device {expected}, but SPARROW-Engine reports {actual}. "
            "Use the detector wheel matching the requested device."
        )


def _predict(args):
    from wildquest.inference import (
        EfficientNetClassifier, InferenceError, InferencePipeline, PhotoInferenceResult,
    )

    def failure(stage, error, code, oriented_size=None):
        issue = InferenceError(stage, type(error).__name__, str(error))
        return PhotoInferenceResult("error", oriented_size, errors=(issue,)), code

    try:
        image = _image_path(args.image)
    except (OSError, ValueError) as exc:
        return failure("input", exc, 2)

    try:
        bundle = args.model_bundle.resolve()
        read_model_bundle(bundle)
    except (OSError, ValueError) as exc:
        return failure("detector", exc, 2)

    try:
        import sparrow_engine

        if args.detector_device.startswith("cuda:"):
            import onnxruntime

            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is unavailable. Use the GPU detector runtime.")
        sparrow_engine.init(device=args.detector_device, model_dir=str(bundle.parent))
        _check_detector_device(sparrow_engine, args.detector_device)
        detector = MegaDetector(confidence_threshold=args.threshold)
    except Exception as exc:
        return failure("detector", exc, 1)

    try:
        classifier = EfficientNetClassifier(args.checkpoint, device=args.classifier_device)
    except (OSError, ValueError) as exc:
        return failure("classifier", exc, 2)
    except Exception as exc:
        return failure("classifier", exc, 1)

    print(f"Detector on {args.detector_device}, classifier on {args.classifier_device}", file=sys.stderr)
    started = perf_counter()
    result = InferencePipeline(detector, classifier).predict(image)
    print(f"Photo inference took {perf_counter() - started:.3f} s", file=sys.stderr)
    if result.status != "error":
        try:
            _check_detector_device(sparrow_engine, args.detector_device)
        except Exception as exc:
            return failure("detector", exc, 1, result.oriented_size)
    if result.status == "error":
        return result, 2 if result.errors[0].stage == "input" else 1
    return result, 3 if result.status == "partial" else 0


def _run(args, output):
    try:
        with redirect_stdout(sys.stderr):
            result, code = _predict(args)
        for error in result.errors:
            crop = "" if error.detection_index is None else f" (detection {error.detection_index})"
            print(f"{error.stage}{crop}: {error.message}", file=sys.stderr)

        payload = json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n"
        if args.output is None:
            output.write(payload)
            output.flush()
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        return code
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"Prediction command failed: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    return _run(parse_args(argv), sys.stdout)


if __name__ == "__main__":
    args = parse_args()
    # Keep native detector and ORT diagnostics off the JSON stream as well.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as result_stream:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.exit(_run(args, result_stream))
