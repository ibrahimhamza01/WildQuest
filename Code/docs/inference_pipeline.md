# Integrated wildlife inference

`predict_image.py` runs MegaDetector V6 (`MDV6-yolov10-e`) and the trained 25-class EfficientNet-B0 in one Python process. The pipeline decodes the photo into EXIF-oriented RGB pixels, detects animals, and classifies each valid crop with the classifier's validation transform. Each prediction retains its detector box and both confidence scores. The classifier checks the checkpoint against the frozen class mapping and dataset manifest hash when it loads.

## Runtime and model files

Run the examples from the repository root using the classifier environment. Verification used Linux x86-64, Python `3.13.14`, PyTorch `2.13.0+cu130`, torchvision `0.28.0+cu130`, and Pillow `12.2.0`. The base Python dependencies are in [requirements.txt](../requirements.txt).

Choose the detector packages for the device that will run MegaDetector. The CPU and GPU distributions share import names. Use separate Python environments if you need both detector variants. The CPU detector can run in a CUDA-enabled PyTorch environment for mixed execution. [SPARROW-Engine installation guide](https://github.com/microsoft/SPARROW-Engine#python-package-only-pypi)

- **CPU detector.** In a classifier environment prepared for the CPU detector, run `python -m pip install -r Code/requirements-inference.txt`. This installs `sparrow-engine==0.1.32` and `onnxruntime==1.25.1`. For a CUDA classifier with this detector, use a CUDA-enabled PyTorch build in the same environment.
- **CUDA detector.** Use the EC2 CUDA-enabled `dl-venv` with `sparrow-engine-gpu==0.1.32` and `onnxruntime-gpu==1.25.1`. On a fresh GPU inference environment with the classifier dependencies installed, add these two versioned packages with `python -m pip install sparrow-engine-gpu==0.1.32 onnxruntime-gpu==1.25.1`.

Run `python -m pip check` after setup. For GPU setup, check that `nvidia-smi` sees the GPU and that Python reports the expected runtime before running the photo example.

```bash
nvidia-smi
python -c 'import onnxruntime as ort, torch; print(torch.cuda.is_available()); print(ort.get_available_providers())'
```

A CUDA classifier requires `True` from PyTorch. A CUDA detector requires `CUDAExecutionProvider` in the provider list. The CLI checks the detector's reported active device and surfaces initialization and inference failures.

The default checkpoint is `outputs/baselines/efficientnet_b0_v1/best.pt`. To use the default training output, pass `--checkpoint outputs/efficientnet_b0/best.pt`. For MegaDetector, point `--model-bundle` to the unpacked `MDV6-yolov10-e` directory containing `manifest.toml`, `labels.txt`, and `1/model.onnx`. The CLI passes the bundle's parent to SPARROW-Engine. [Detector artifact setup and checksums](animal_detection.md#set-up-the-isolated-cpu-environment)

## Run one photograph

Set `detector_bundle` to the bundle you have on disk. This example uses the training photograph from the standalone detector guide. You can also pass your own photograph or a frozen validation photograph. The CLI rejects repository paths into the general and DMV held-out splits.

```bash
detector_bundle=outputs/detection/models/sparrow-zoo-0.33.0/MDV6-yolov10-e

# CPU detector and CPU classifier, with JSON on stdout
python Code/scripts/predict_image.py \
  --image data/raw/images/train/02/18592269.jpg \
  --model-bundle "$detector_bundle"

# CPU detector and CUDA classifier, using the CPU detector packages
python Code/scripts/predict_image.py \
  --image data/raw/images/train/02/18592269.jpg \
  --model-bundle "$detector_bundle" \
  --detector-device cpu --classifier-device cuda:0

# CUDA detector and CUDA classifier, using the GPU detector packages
python Code/scripts/predict_image.py \
  --image data/raw/images/train/02/18592269.jpg \
  --checkpoint outputs/baselines/efficientnet_b0_v1/best.pt \
  --model-bundle "$detector_bundle" --threshold 0.20 \
  --detector-device cuda:0 --classifier-device cuda:0 \
  --output outputs/prediction.json
```

| Argument | Behavior |
| --- | --- |
| `--image PATH` | Required local photograph. |
| `--model-bundle DIR` | Required unpacked detector bundle. |
| `--checkpoint PATH` | Local checkpoint. Defaults to the baseline above. |
| `--threshold FLOAT` | Animal detection confidence cutoff from 0 to 1. Defaults to `0.20`. |
| `--detector-device DEVICE` | `cpu`, `cuda`, or `cuda:N`. Defaults to `cpu`. `cuda` selects `cuda:0`. |
| `--classifier-device DEVICE` | Same device syntax and default. |
| `--output PATH` | Write JSON to a new file in an existing directory. When supplied, stdout is empty. Existing files are protected from overwrite. |

The CLI sends diagnostics, active device information, and the photo inference time to stderr. For a compact check of the output file, run `python -m json.tool outputs/prediction.json`.

## JSON response and exit codes

The CLI serializes the reusable pipeline's `PhotoInferenceResult.to_dict()` using strict JSON numbers. The following is an illustrative single-animal response.

```json
{
  "status": "ok",
  "oriented_size": [1024, 768],
  "predictions": [
    {
      "box_xyxy": [735, 274, 991, 461],
      "category": "animal",
      "detector_confidence": 0.74,
      "class_id": 2,
      "scientific_name": "Branta canadensis",
      "classification_confidence": 0.88,
      "top3": [
        {"class_id": 2, "scientific_name": "Branta canadensis", "confidence": 0.88},
        {"class_id": 0, "scientific_name": "Anas platyrhynchos", "confidence": 0.07},
        {"class_id": 6, "scientific_name": "Chrysemys picta", "confidence": 0.03}
      ]
    }
  ],
  "errors": []
}
```

`oriented_size` is `[width, height]` in source pixels after EXIF correction. It is `null` if the pipeline has no decoded dimensions. Boxes are integer `[left, top, right, bottom]` coordinates in that pixel grid, with right and bottom exclusive. Predictions stay in detector order, including overlapping boxes. The highest-ranked of the three species is also the primary prediction. Species IDs range from 0 to 24, and confidence scores remain numeric at model precision.

| Status | Exit code | Meaning |
| --- | --- | --- |
| `ok` | `0` | All detected animals were classified. |
| `no_detection` | `0` | Detection completed and `predictions` is empty. |
| `partial` | `3` | At least one prediction succeeded and at least one detection, crop, or classification failed. |
| `error` | `2` | The photo or a model artifact was missing or invalid. |
| `error` | `1` | A fatal runtime or inference error occurred. |

Every error entry contains `stage` (`input`, `detector`, `crop`, or `classifier`), `error_type`, `message`, and `detection_index`. The index is zero-based in detector order and is `null` for failures that apply to the entire photo or setup. Argument syntax errors exit `2` with argparse diagnostics on stderr. Failure to write the requested output exits `1` with a stderr diagnostic.

## Verification and limitations

```bash
python -m unittest discover -s Code/tests -p test_inference.py -v
python -m unittest discover -s Code/tests -v
```

The automated tests use synthetic images and mocked models for CLI and pipeline cases. The real smoke test on November 27, 2026 used the baseline checkpoint, the Sparrow Model Zoo `0.33.0` ONNX bundle, and training photo `18592269`. CPU, mixed CPU/CUDA, and full CUDA all produced seven valid boxes and the same ranked class IDs. Against CPU, the largest full-CUDA difference was about `0.00076` in detector confidence and `0.00520` among the reported classifier probabilities. An Nsight Systems CUDA trace on the NVIDIA A10G with driver `595.99.02` recorded convolution kernels for both models. SPARROW-Engine reported CPU decoding of the temporary lossless PNG during the CUDA run.

| Devices | Single CLI wall time | Photo inference time |
| --- | --- | --- |
| CPU detector and CPU classifier | 10.02 s | 5.177 s |
| CUDA detector and CUDA classifier | 6.40 s | 2.500 s |

The CLI wall time includes startup and model initialization. The photo inference timer begins after initialization and includes the detector's first model load, cropping, and all seven classifier calls. In separate instrumented runs, the detector and all classifier calls took 6.287 s and 0.203 s on CPU, and 1.251 s and 0.388 s on CUDA. These are single-photo observations.

The CLI handles one photograph per process. Classification depends on the animal crops selected by MegaDetector. Detector and classifier confidences are scores from different models, and the species probabilities are uncalibrated.
