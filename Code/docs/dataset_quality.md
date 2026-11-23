# Dataset Quality and Augmentation (Issue #4)

This audit checks the frozen WildQuest manifest and class mapping and inspects local
training images. It recommends an EfficientNet-B0 augmentation experiment.

## How to reproduce

From the repository root, use the course Python 3.13 environment. See the
[environment setup](../README.md#environment) for the CUDA-compatible PyTorch build.

```bash
python -m pip install -r Code/requirements.txt
python Code/scripts/audit_dataset.py           # manifest statistics for all splits
python Code/scripts/audit_dataset.py --images  # also decode local training images
python -m unittest discover -s Code/tests -p 'test_dataset_audit.py' -v
```

The image audit uses the existing download. It reports eligible training rows, local
files found, files inspected, unreadable files, and missing files. The `val`, `test`,
and `dmv_test` splits contribute manifest metadata only. The separate integrity check
below reads file bytes for hashes without decoding images.

Outputs are local and Git-ignored under `outputs/dataset_audit/`. Pass `--out` with
a fresh directory there to keep outputs from different runs distinct.

- `summary.txt`
- `class_distribution.csv` and `class_distribution.png`
- `observations_per_class.csv`
- `resolution.csv`
- `duplicate_files.csv`
- `image_inspection.csv` when `--images` is used
- `species_grid.png` and `augmentations/*.png` when readable training images are available

## 1. Manifest-level findings (all 25,000 images)

| Check | Result |
|---|---|
| Rows / species / observations | 25,000 / 25 / 25,000 |
| `class_mapping.json` vs manifest | 25 species, 0 mismatched class IDs |
| Images per species | exactly 1,000 for every species (max/min ratio 1.00) |
| Split sizes | train 18,750 (75%), val 2,500 (10%), test 2,500 (10%), dmv_test 1,250 (5%) |
| Photos per observation | exactly 1 |
| Observations appearing in more than one split | 0 |
| Duplicate SHA-256 hashes in the manifest | 0 |
| Animal groups | Bird 11,000 (11 species), Mammal 8,000 (8), Reptile 4,000 (4), Amphibian 2,000 (2) |
| Licenses | CC-BY 19,571 (78%), CC0 5,429 (22%) |
| DMV-located share | dmv_test 100%, train 8.3%, val 5.2%, test 0% |
| Short side (px) | min 256, median 768, mean 738, max 1024 |
| Short side below 224 px | 0 |
| Aspect ratio strictly above 2:1 | 490 (about 2%) |

- Each class has 750 training images. Class weighting is unnecessary for this split.
  Birds account for 11 of the 25 species, so per-class accuracy remains useful.
- Each observation ID occurs in exactly one split.
- Every source image has a short side of at least 256 px. Evaluation resize needs
  no upscaling, although a training crop can produce a smaller region that is upscaled
  to 224 px.
- A square center crop can discard substantial content from the 490 wide or tall
  images (363 in `train`). `resolution.csv` flags them using the unrounded ratio.
- Preserve the photo's license and credit the photographer when sharing CC-BY source
  images or derived figures publicly. Keep audit figures in the Git-ignored output
  directory.

### Separate file integrity check

All 25,000 original dataset files were independently verified against the frozen
manifest's SHA-256 hashes. To repeat the hash check on an existing local copy, run
`python Code/scripts/get_dataset.py --verify-only`. This check is separate from the
training-only image audit.

## 2. Image-level findings

### Full training audit

The audit selected all rows with `split == "train"`. It decoded each available
image with Pillow, compared its native dimensions to the manifest, and recorded
color mode, EXIF orientation and mean grayscale brightness. The species grid uses
up to the first six readable training images per species in manifest order. Each
augmentation figure uses one readable training image per animal group, selected
in class ID and photo ID order.

| Check | Independently verified result |
|---|---|
| Training images inspected | 18,750 of 18,750 |
| Unreadable images | 0 |
| Size differs from manifest | 0 |
| Color modes | RGB 18,521, L 176, RGBA 51, P 1, LA 1 |
| Very dark images (mean brightness < 40) | 159 |
| Very bright images (mean brightness > 215) | 75 |
| Species grid | 1 grid representing all 25 species |
| Augmentation examples | 4 figures, one per animal group |

- **Color mode.** Convert inputs to RGB so that grayscale, palette and alpha-channel
  images all produce the three channels expected by EfficientNet and can be batched
  together.
- **Small subjects and background.** The figures show animals occupying small parts
  of the frame, such as a duck surrounded by water or a bird among branches. Aggressive
  crops can remove these subjects.
- **Exposure.** The dark and bright images support testing mild brightness and contrast
  augmentation.

### Original 500-image training sample

The initial audit selected 20 random training images per species (`random_state=0`)
from the manifest using iNaturalist `photo_url` values. All 500 sample hashes matched
the manifest. The following results describe that original sample.

| Check | Original sample (n = 500) |
|---|---|
| Unreadable or corrupt files | 0 |
| Size differs from manifest | 0 |
| EXIF-rotated images | 0 |
| Grayscale images (mode `L`) | 6 (1.2%) |
| Very dark images (mean brightness < 40) | 4 |
| Very bright images (mean brightness > 215) | 2 |

## 3. Recommended preprocessing and augmentation

The classifier currently uses a bicubic training crop with `scale=(0.75, 1.0)`
and horizontal flipping. The candidate training recipe uses a wider crop scale
and adds color jitter. The evaluation recipe uses the same weight preset as the
classifier's validation pipeline.

```python
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

evaluation = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
training = transforms.Compose([
    transforms.RandomResizedCrop(
        evaluation.crop_size, scale=(0.6, 1.0), interpolation=evaluation.interpolation,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=evaluation.mean, std=evaluation.std),
])

# Convert a source image to RGB before applying training or evaluation.
```

The figures show the original RGB image and the preset's bicubic resize of the
short side to 256 px followed by a 224 × 224 center crop. The standalone random
resized crop uses the original image. Forced horizontal and vertical flips, color
jitter and rotation each use the same evaluation crop as their starting image. The
last panel shows the candidate training sequence of random resized crop, horizontal
flip at probability 0.5 and color jitter. The script uses torchvision for each
panel, with a CPU random seed of 0 for each figure. The panels show pixels before
tensor conversion and normalization.

| Setting | Recommendation | Reason |
|---|---|---|
| Input resolution | 224 × 224 | Matches the pretrained EfficientNet-B0 input crop. |
| Normalization | ImageNet mean/std | Matches the pretrained weights. |
| RGB conversion | Always | Standardizes input channels across all five observed color modes. |
| RandomResizedCrop | `scale=(0.6, 1.0)` | The higher minimum sampled area reduces the chance of cropping out small subjects while allowing zoom and position variation. Torchvision can fall back to a centered crop when no sampled crop fits. |
| Horizontal flip | p = 0.5 | Animals appear facing either direction; a flip never changes the species. |
| ColorJitter | 0.3 brightness/contrast/saturation, **no hue** | Handles lighting variety (dark and bright images exist). Hue is left at 0 because color identifies several species (Northern Cardinal, Blue Jay, Green Frog). |
| Rotation | Omit by default | The ±15° illustration can introduce black corners. |
| Vertical flip | Do not use | Produces unnatural images (animals upside down, water above land). |
| Class balancing | None needed | Every species has 750 training images. |

## 4. Suggested follow-ups

1. If accuracy is limited by small subjects, compare training on MegaDetector crops
   (#5) against full images, using the same augmentation.
2. Use validation errors (#6) to check whether center cropping removes subjects in
   the 45 validation images with aspect ratios above 2:1.
