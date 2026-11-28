# Animal detection and cropping

WildQuest locates animals in full photographs so a later species-classification step can work with crops of the subjects.

**Full photograph → animal detection → bounding boxes → animal crops**

The detector runs as a standalone component. The EfficientNet species classifier has its own environment, training workflow, and checkpoint. MegaDetector supplies a pretrained animal-localization model developed for camera-trap imagery. Using its published weights avoids a separate training effort and the need to annotate WildQuest bounding boxes. This evaluation measures how well the pretrained detector transfers to WildQuest photographs. [MegaDetector model zoo](https://microsoft.github.io/MegaDetector/model_zoo/)

## Model and runtime

The implementation uses Microsoft MegaDetector V6 `MDV6-yolov10-e`, the YOLOv10 Extra variant. The published ONNX bundle uses a 1280 × 1280 letterboxed model input. WildQuest uses SPARROW-Engine `0.1.32` through its Python API with ONNX Runtime `1.25.1`. The initial animal confidence threshold is `0.20`. [SPARROW-Engine Python API](https://pypi.org/project/sparrow-engine/0.1.32/) · [MegaDetector model zoo](https://microsoft.github.io/MegaDetector/model_zoo/)

The weights are licensed **AGPL-3.0** and come from Microsoft's **Sparrow Model Zoo `0.33.0`**, Zenodo record **`23992557`**. Use the individual `camera_trap__detector__MDV6-yolov10-e.zip` archive for this revision. The archive MD5 and packaged file SHA-256 values were checked against the release's published checksums. [Official release](https://zenodo.org/records/23992557) · [Model archive](https://zenodo.org/records/23992557/files/camera_trap__detector__MDV6-yolov10-e.zip)

| Artifact | Algorithm | Checksum |
|---|---|---|
| Model archive | MD5 | `693e5d5d3e991375a7974b8d5f0a7caf` |
| Model archive | SHA-256 | `3f26c3c9c0c16e42e335abfa03211da4053dc4cab8af3c54c1b7bf9625aabf05` |
| `MDV6-yolov10-e/1/model.onnx` | SHA-256 | `9a9f22b85046496fb3b8cdb9b8ad27a4485bf6daf1f43de7fb10f9b44e702c13` |
| `MDV6-yolov10-e/manifest.toml` | SHA-256 | `56ecf2dc6f43442d8710ea4d3d4b239b3032373fe6b85298a5befede3bb207fe` |

The standalone smoke test and validation run used **CPU inference on x86-64 Linux** in a separate **Python `3.13.14`** virtual environment. The standalone CLI initializes the CPU device explicitly. See the [integrated inference guide](inference_pipeline.md) for CUDA setup and verification.

## Implementation and Python API

| File | Responsibility |
|---|---|
| [detection/__init__.py](../src/wildquest/detection/__init__.py) | Exports `MegaDetector`, `Detection`, and `crop_animal`. |
| [megadetector.py](../src/wildquest/detection/megadetector.py) | Lazily imports SPARROW-Engine, invokes the local model, and returns WildQuest detection records. |
| [crops.py](../src/wildquest/detection/crops.py) | Prepares EXIF-oriented RGB images, validates boxes, and extracts crops. |
| [test_detector.py](../scripts/test_detector.py) | Selects inputs, runs detection, and writes results, crops, previews, and coverage summaries. |
| [test_detection.py](../tests/test_detection.py) | Exercises the adapter, crop utilities, and CLI with synthetic inputs and mocked inference. |

`MegaDetector.detect(image)` accepts a string path, a `pathlib.Path`, or an in-memory `PIL.Image.Image`. File inputs use Pillow decoders, including JPEG and PNG. Inference and cropping use the same RGB pixel grid after applying EXIF orientation. The adapter supplies SPARROW-Engine with a lossless temporary PNG and removes it when inference exits, including on failure.

Each backend-independent `Detection` contains `category`, `confidence`, and `box`. The adapter clips the backend's normalized box to the image bounds, then scales x by the oriented image width and y by the oriented height. Lower coordinates round down and upper coordinates round up. The resulting integer `(left, top, right, bottom)` box has an upper-left origin and exclusive right and bottom edges. Malformed, non-finite, and zero-area boxes are discarded.

WildQuest retains every valid `animal` detection returned by the backend at or above the confidence threshold, in backend order. The bundle's `person` and `car` categories are filtered out. Each qualifying box yields one crop at source resolution. Overlapping boxes are retained. A successful image with no qualifying detections returns `[]`. `crop_animal(image, box)` returns an RGB Pillow image for a valid box and `None` for an invalid one. Decode and inference failures propagate to Python callers.

## Set up the isolated CPU environment

Run the commands from the repository root. Use a Python `3.13.14` interpreter for the verified setup and create a virtual environment for detection outside the checkout. The command pins the tested inference packages, including transitive dependencies. The verification environment used pip `26.1.2`.

```bash
detector_venv="$HOME/.venvs/wildquest-detection"
python3.13 --version
python3.13 -m venv "$detector_venv"
detector_python="$detector_venv/bin/python"

"$detector_python" -m pip install --only-binary=:all: \
  sparrow-engine==0.1.32 onnxruntime==1.25.1 Pillow==12.2.0 \
  numpy==2.5.5 flatbuffers==25.12.19 packaging==26.3 protobuf==7.36.2
"$detector_python" -m pip check
```

Keep the following shell variables for the remaining commands. The bundle directory contains `manifest.toml`, `labels.txt`, and `1/model.onnx`. SPARROW-Engine uses the bundle's parent as its model directory.

```bash
detector_model_dir="$PWD/outputs/detection/models/sparrow-zoo-0.33.0"
detector_bundle="$detector_model_dir/MDV6-yolov10-e"
(
  set -e
  mkdir -p "$detector_model_dir"
  cd "$detector_model_dir"
  detector_archive=camera_trap__detector__MDV6-yolov10-e.zip
  if [ ! -f "$detector_archive" ]; then
    curl --fail --location --remove-on-error --output "$detector_archive" \
      "https://zenodo.org/records/23992557/files/$detector_archive"
  fi
  printf '%s  %s\n' '693e5d5d3e991375a7974b8d5f0a7caf' "$detector_archive" | md5sum -c -
  printf '%s  %s\n' '3f26c3c9c0c16e42e335abfa03211da4053dc4cab8af3c54c1b7bf9625aabf05' \
    "$detector_archive" | sha256sum -c -
  unzip -n "$detector_archive"
  sha256sum -c <<'CHECKSUMS'
9a9f22b85046496fb3b8cdb9b8ad27a4485bf6daf1f43de7fb10f9b44e702c13  MDV6-yolov10-e/1/model.onnx
56ecf2dc6f43442d8710ea4d3d4b239b3032373fe6b85298a5befede3bb207fe  MDV6-yolov10-e/manifest.toml
CHECKSUMS
)
```

### Run from Python

Set the model directory before the first inference call. This example uses a training photo from the frozen dataset. You can also supply your own photograph.

```bash
PYTHONPATH=Code/src SPARROW_ENGINE_DEVICE=cpu \
  SPARROW_ENGINE_MODEL_DIR="$detector_model_dir" "$detector_python" - <<'PY'
from pathlib import Path

from PIL import Image
from wildquest.detection import MegaDetector, crop_animal

photo = Path("data/raw/images/train/02/18592269.jpg")
detector = MegaDetector(confidence_threshold=0.20)
with Image.open(photo) as image:
    for detection in detector.detect(image):
        crop = crop_animal(image, detection.box)
        if crop is not None:
            print(detection.category, detection.confidence, detection.box, crop.size)
PY
```

### Run the standalone CLI

Use `--image` with one or more paths, or `--split` with `train` or `val`. Manifest sampling defaults to four images per class and seed `42`. Explicit paths can refer to user-supplied images. Paths inside the repository dataset must match the frozen train/validation manifest, and the CLI verifies the selected image hashes. The general and DMV test splits are held out. Choose a new `--output-dir` under `outputs/detection/` for each run.

```bash
"$detector_python" Code/scripts/test_detector.py \
  --image data/raw/images/train/02/18592269.jpg \
  --model-bundle "$detector_bundle" \
  --output-dir outputs/detection/single-photo

"$detector_python" Code/scripts/test_detector.py \
  --split val --sample-per-class 4 --seed 42 --threshold 0.20 \
  --model-bundle "$detector_bundle" \
  --output-dir outputs/detection/val-42
```

The CLI writes the following artifacts in the run directory:

- `results.json` records the model hashes, engine version, device, threshold, manifest hash, and selection parameters. Each image record contains its source, available dataset metadata, original and oriented dimensions, inference status, detection count, boxes, confidence scores, and run-relative artifact paths. Errors include the processing stage and message.
- `summary.json` reports image counts and percentages overall, by wildlife group, and by manifest species. Detection percentages use images with successful inference. Processing failure percentages use images attempted.
- `crops/` contains one PNG per qualifying detection. `previews/` contains JPEGs with boxes and confidence scores. An empty detection result still gets a preview.

The CLI exits with a nonzero status if any input fails processing and retains its report. The model bundle and generated artifacts under `outputs/detection/` are Git-ignored. Following the setup and run commands regenerates the automated outputs. The original evaluation's `manual-review.json` contains separate visual judgments and can be recreated by reviewing the previews.

### Run tests

Run the focused detector tests in the detector environment. Run the full suite with the project's classifier development environment active. The tests use mocked detection and synthetic inputs.

```bash
"$detector_python" -m unittest discover -s Code/tests -p test_detection.py -v
python -m unittest discover -s Code/tests -v
```

## Validation results

The experiment used only the frozen `val` split. The sampler visited the 25 class IDs in order, sorted each class's candidates lexicographically by photo ID, and shuffled each class with a shared `random.Random(42)`. It preferred unused observation UUIDs and selected four photos per class. The resulting 100 photos came from **100 distinct observations**. Each sampled file matched its manifest checksum. The [frozen manifest checksum](../configs/dataset_manifest.sha256) was:

```text
077add77be30fd2766e3b955b04eec3b2c7dd5f737adba50cd7b8b1474e2d9e9
```

CPU inference at confidence `0.20` processed **100/100** images successfully. **86/100** had at least one qualifying animal box, **14/100** had zero, and **27/100** had multiple boxes. The run produced **155 crops** and **100 previews**.

| Wildlife group | Images with a detection | Coverage |
|---|---:|---:|
| Birds | 40/44 | 90.91% |
| Mammals | 27/32 | 84.38% |
| Reptiles | 13/16 | 81.25% |
| Amphibians | 6/8 | 75.00% |
| **Overall** | **86/100** | **86.00%** |

| Manifest species | Images with a detection | Coverage |
|---|---:|---:|
| *Anas platyrhynchos* | 4/4 | 100% |
| *Ardea herodias* | 4/4 | 100% |
| *Branta canadensis* | 4/4 | 100% |
| *Buteo jamaicensis* | 3/4 | 75% |
| *Cardinalis cardinalis* | 4/4 | 100% |
| *Chelydra serpentina* | 2/4 | 50% |
| *Chrysemys picta* | 4/4 | 100% |
| *Corvus brachyrhynchos* | 4/4 | 100% |
| *Cyanocitta cristata* | 3/4 | 75% |
| *Didelphis virginiana* | 2/4 | 50% |
| *Lithobates catesbeianus* | 3/4 | 75% |
| *Lithobates clamitans* | 3/4 | 75% |
| *Marmota monax* | 4/4 | 100% |
| *Odocoileus virginianus* | 4/4 | 100% |
| *Pantherophis alleghaniensis* | 3/4 | 75% |
| *Passer domesticus* | 3/4 | 75% |
| *Plestiodon fasciatus* | 4/4 | 100% |
| *Procyon lotor* | 1/4 | 25% |
| *Sciurus carolinensis* | 4/4 | 100% |
| *Sturnus vulgaris* | 3/4 | 75% |
| *Sylvilagus floridanus* | 4/4 | 100% |
| *Tamias striatus* | 4/4 | 100% |
| *Turdus migratorius* | 4/4 | 100% |
| *Vulpes vulpes* | 4/4 | 100% |
| *Zenaida macroura* | 4/4 | 100% |

These figures measure **image-level coverage**. The frozen manifest supplies image-level labels, so object-level recall, precision, and mAP would require ground-truth boxes. Species-classification accuracy requires a separate classifier evaluation. With four photos per species, a single result changes the species coverage by 25 percentage points.

## Error and limitation analysis

Visual review of all 14 zero-detection previews found **10 with a visible animal**, **3 with an ambiguous or indistinct target**, and **1 showing tracks without a visible animal**. The last case was a raccoon-labeled photo (photo ID `702706110`), where the tracks provide indirect evidence. The ambiguity in the other three cases limits conclusions about detector misses.

- **Small or distant subjects.** A small flying bird against the sky received no box (`380309604`). Multiple distant geese received boxes in a flock photo (`243036752`), whose individual animals were left unannotated.
- **Reptiles and amphibians.** Two small turtles on a log received separate boxes (`239701529`), and a visible frog in shallow water received one box (`10537412`). A clearly visible snapping turtle (`419321423`) and a partly submerged bullfrog (`18863867`) received zero boxes.
- **Overlapping boxes.** The automatic geometry check found **16 images** with a positive-area box intersection, including **6** with at least one pair at intersection over union (IoU) **≥ 0.50**. Two large boxes covered the same visible frog (`30589176`).
- **Boxes spanning several animals.** Overlapping boxes included parts of both a deer and its fawn (`227516634`). Individual boxes can duplicate a subject, cover only part of one, or contain several.
- **Crop correspondence.** The highest-confidence box in a robin-labeled photo covered mostly branches and part of the bird. A smaller box followed the visible subject (`588751522`). The animal score reflects the detector's confidence in the box category. The labeled species and the suitability of its crop require separate review.

The manual review examined every zero-detection image and selected positives. Systematic crop suitability across all 100 images remains unmeasured. The observations and overlap counts describe this small validation sample.
