# EfficientNet-B0 Classifier

WildQuest's species classifier uses transfer learning to identify 25 DMV wildlife species from photographs.

## Architecture

- Framework: PyTorch / torchvision
- Model: EfficientNet-B0
- Pretrained weights: `EfficientNet_B0_Weights.IMAGENET1K_V1`
- Input: 224 × 224 RGB images
- Output: 25 class logits
- Loss: Cross-entropy
- Optimizer: AdamW

The original 1,000-class ImageNet classification layer is replaced with a new 25-class linear layer.

## Implementation

| File | Responsibility |
|---|---|
| `dataset.py` | Read frozen manifest, load images, apply transforms, create DataLoaders |
| `model.py` | Construct EfficientNet, freeze/unfreeze layers, configure optimizer |
| `../../../../scripts/train_classifier.py` | Execute training, validation, and checkpointing |

Training configuration: `Code/configs/train_efficientnet_b0.json`.

### Transfer Learning

Training uses two stages:

1. **Head training (3 epochs):** Freeze the pretrained feature extractor and train only the new classification head.
2. **Fine-tuning (up to 12 epochs):** Unfreeze the final two EfficientNet feature blocks while keeping the earlier blocks frozen.

The classifier head uses learning rate `0.001`, while unfrozen backbone parameters use `0.0001`. Weight decay is `0.0001`.

Training uses a batch size of 32, random resized cropping, horizontal flipping, ImageNet normalization, and CUDA automatic mixed precision. Validation uses the pretrained weights' deterministic preprocessing.

The best checkpoint is selected by validation accuracy, with validation loss used to break ties. Early-stopping patience is four non-improving epochs per training stage.

## Baseline Results

First training run on AWS EC2 `g5.xlarge` with NVIDIA A10G:

| Metric | Result |
|---|---|
| Training images | 18,750 |
| Validation images | 2,500 |
| Completed epochs | 15 |
| Best validation accuracy | **82.04%** |
| Best checkpoint epoch | 11 |
| Validation loss at epoch 11 | 0.6635 |
| Lowest validation loss | 0.5929 (epoch 6) |

Training loss continued decreasing while validation loss increased after approximately epoch 6, indicating overfitting.

The saved checkpoint corresponds to epoch 11 under the validation-accuracy selection criterion.

These results are from validation, not the held-out test sets.

## Artifacts

Default output directory: `outputs/efficientnet_b0/`

- `best.pt` — Selected model checkpoint, training configuration, class mapping, optimizer state, and dataset manifest checksum.
- `history.csv` — Per-epoch training and validation metrics.

The initial baseline is also preserved locally in `outputs/baselines/efficientnet_b0_v1/`.

Checkpoints are excluded from Git. The current training script does not support exact interrupted-run resumption.

## Limitations and Next Steps

- Optimize training throughput on the EC2 instance.
- Investigate overfitting using validation data.
- Perform final general and DMV test-set evaluation through the separate evaluation pipeline.
- Integrate the selected classifier with the animal-detection and application pipelines.