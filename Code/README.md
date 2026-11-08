# Dataset

25 wildlife species; 25,000 licensed iNaturalist photographs. Fixed selection from the 2026-10-27 Open Dataset snapshot. The manifest includes class labels, observation IDs, licenses, original photo URLs, splits and SHA-256 hashes.

## Download

Install Python 3.9+ and AWS CLI v2. You need read access to the team S3 bucket.

```bash
python Code/scripts/get_dataset.py
```

The script downloads into `data/raw/images/`, verifies all 25,000 hashes, and can repair missing files. To check an existing local dataset without S3 access:

```bash
python Code/scripts/get_dataset.py --verify-only
```

Source priority: `--source`, `WILDQUEST_DATASET_URI`, then `DEFAULT_DATASET_URI` in `Code/scripts/get_dataset.py`. The maintainer sets the default once before merging.

## Integrity

The committed `Code/configs/dataset_manifest.csv` is frozen. Don't modify the class map or test splits. Check integrity with `python -m unittest discover -s Code/tests -v`.

CC-BY images require attribution. Before redistributing images or using them publicly, verify the photographer/creator credits; observation author IDs alone may not be sufficient. DMV coverage is an approximate geographic bounding box.
