# WildQuest

WildQuest identifies common wildlife photographed in Washington, D.C., Maryland and Northern Virginia. It will use a pretrained animal detector, an EfficientNet species classifier, and an observation/XP system.

## Dataset

25 classes; 25,000 validated, licensed iNaturalist photographs. See [dataset setup](Code/README.md).

```bash
python Code/scripts/get_dataset.py
```

Requires Python 3.9+, AWS CLI v2 and read access to the team dataset S3 bucket.

## Development

Issue → branch → PR → review → merge. Do not commit raw images, model weights, or credentials.
