# Dataset Quality and Augmentation (Issue #4)

This audit covers the frozen WildQuest dataset (`Code/configs/dataset_manifest.csv`,
`Code/configs/class_mapping.json`). It recommends preprocessing and augmentation
settings for the EfficientNet classifier. The manifest, class IDs and splits were
not modified.

## How to reproduce

```bash
python3 Code/scripts/audit_dataset.py            # manifest-level checks
python3 Code/scripts/audit_dataset.py --images   # also inspect downloaded images
python3 -m unittest discover -s Code/tests -p "test_dataset_audit.py" -v
```

Outputs are written to `outputs/dataset_audit/`:
- `summary.txt`
- `class_distribution.csv` and `class_distribution.png`
- `observations_per_class.csv`
- `resolution.csv`
- `duplicate_files.csv`
- `image_inspection.csv`
- `species_grid.png`
- `augmentations/*.png`

## 1. Manifest-level findings (all 25,000 images)

| Check | Result |
|---|---|
| Rows / species / observations | 25,000 / 25 / 25,000 |
| `class_mapping.json` vs manifest | 25 species, 0 mismatched class IDs |
| Images per species | exactly 1,000 for every species (max/min ratio 1.00) |
| Split sizes | train 18,750 (75%), val 2,500 (10%), test 2,500 (10%), dmv_test 1,250 (5%) |
| Photos per observation | exactly 1 |
| Observations appearing in more than one split | 0 |
| Byte-identical duplicate images (SHA-256) | 0 |
| Animal groups | Bird 11,000 (11 species), Mammal 8,000 (8), Reptile 4,000 (4), Amphibian 2,000 (2) |
| Licenses | CC-BY 19,571 (78%), CC0 5,429 (22%) |
| DMV-located share | dmv_test 100%, train 8.3%, val 5.2%, test 0% |
| Short side (px) | min 256, median 768, mean 738, max 1024 |
| Short side below 224 px | 0 |
| Aspect ratio above 2:1 | 488 (about 2%) |

**What this means:**
- **Balance.** Classes are perfectly balanced, so the training pipeline does not need
  class weights or a weighted sampler. The dataset is bird-heavy by species count
  (11 of 25), so per-class metrics should still be reported alongside accuracy.
- **Leakage.** Every observation contributes one photo and none crosses splits. Train
  and test therefore cannot contain the same encounter.
- **Resolution.** Every image is larger than EfficientNet-B0's 224 px input, so no
  image needs upscaling.
- **Very wide or tall images.** The 488 images above 2:1 lose a lot of content under a
  square center crop. They are listed in `resolution.csv` (`extreme_aspect = True`).
- **Licensing.** CC-BY images require photographer attribution if they are shown
  publicly (report, presentation, app). For that reason this audit does not commit
  any image-derived figures to the public repo.

## 2. Image-level findings (500-image sample)

Students cannot read the team S3 bucket (the course account has an explicit deny),
so image-level checks were run on a sample instead of the full dataset. The sample
was 20 random training images per species (`random_state=0`), downloaded from the
manifest's `photo_url` (iNaturalist open data). **All 500 matched the manifest's
SHA-256 hashes**, so they are identical to the team copies. To repeat these checks on
the full dataset, run `--images` after `get_dataset.py`.

| Check | Result (n = 500) |
|---|---|
| Unreadable or corrupt files | 0 |
| Size differs from manifest | 0 |
| EXIF-rotated images | 0 |
| Grayscale images (mode `L`) | 6 (1.2%) |
| Very dark images (mean brightness < 40) | 4 |
| Very bright images (mean brightness > 215) | 2 |

**What this means:**
- **Grayscale images.** About 1% of images are single-channel. The data loader **must**
  call `.convert("RGB")` on every image. Otherwise these images break batching or
  feed EfficientNet the wrong number of channels. Extrapolated to the full dataset,
  that is roughly 300 images.
- **Small subjects and background.** The species grid and augmentation examples show
  that many animals occupy a small part of the frame. Typical cases are a duck in a
  large area of water, or a bird in branches. This is the main quality concern. It
  affects how aggressive cropping can be (see below), and it supports the planned
  MegaDetector cropping step (#5).
- **Exposure.** A few very dark (night) and very bright images exist. This is normal
  for wildlife photos and supports mild brightness/contrast augmentation.

## 3. Recommended preprocessing and augmentation

Written as torchvision transforms for the EfficientNet training pipeline (#3).

**Training:**
```python
transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Validation / test:**
```python
transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

| Setting | Recommendation | Reason |
|---|---|---|
| Input resolution | 224 x 224 | Native EfficientNet-B0 size; all images are at least 256 px, so nothing is upscaled. |
| Normalization | ImageNet mean/std | Matches the pretrained weights. |
| RGB conversion | Always | About 1% of images are grayscale. |
| RandomResizedCrop | `scale=(0.6, 1.0)` | The torchvision default (0.08, 1.0) often crops small animals out of the frame entirely. 0.6 still adds zoom and position variety. |
| Horizontal flip | p = 0.5 | Animals appear facing either direction; a flip never changes the species. |
| ColorJitter | 0.3 brightness/contrast/saturation, **no hue** | Handles lighting variety (dark and bright images exist). Hue is left at 0 because color identifies several species (Northern Cardinal, Blue Jay, Green Frog). |
| Rotation | Not recommended by default (at most ±10° if tried) | Even ±15° leaves black wedges in the corners after cropping (see `augment_*.png`). The gain is small for upright wildlife photos. |
| Vertical flip | Do not use | Produces unnatural images (animals upside down, water above land). |
| Class balancing | None needed | Every species has exactly 1,000 images. |

## 4. Suggested follow-ups

1. Run `audit_dataset.py --images` on the full 25,000 images once team members can
   read the S3 bucket, to confirm the sample results.
2. If accuracy is limited by small subjects, compare training on MegaDetector crops
   (#5) against full images, using the same augmentation.
3. Check per-class errors (#6) on the 488 extreme-aspect images, since center-cropping
   may cut the animal out at evaluation time.