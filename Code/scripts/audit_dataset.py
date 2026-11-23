"""Audit the frozen WildQuest dataset (issue #4).

Reads Code/configs/dataset_manifest.csv (read-only) and writes summaries to
outputs/dataset_audit/. Never modifies the manifest or splits.

Usage:
    python3 Code/scripts/audit_dataset.py
    python3 Code/scripts/audit_dataset.py --images   # inspect training images and save figures
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "Code" / "configs" / "dataset_manifest.csv"
DEFAULT_CLASS_MAP = REPO_ROOT / "Code" / "configs" / "class_mapping.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "dataset_audit"

SMALL_SIDE_PX = 224        # EfficientNet-B0 input size; smaller images must be upscaled
EXTREME_ASPECT = 2.0       # long side / short side above this is unusual
AUGMENTATION_SEED = 0


def load_manifest(path):
    return pd.read_csv(path)


def load_class_mapping(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def class_distribution(m):
    """Photos per species per split, with a total column."""
    table = m.pivot_table(index=["class_id", "common_name"], columns="split",
                          values="photo_id", aggfunc="count", fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def observation_distribution(m):
    """Distinct observations (encounters) per species per split."""
    return m.pivot_table(index=["class_id", "common_name"], columns="split",
                         values="observation_uuid", aggfunc="nunique", fill_value=0)


def photos_per_observation(m):
    return m.groupby("observation_uuid").size().describe()


def leakage_check(m):
    """Observations that appear in more than one split (should be none)."""
    n_splits = m.groupby("observation_uuid")["split"].nunique()
    return n_splits[n_splits > 1].index.tolist()


def duplicate_files(m):
    """Rows whose image bytes are identical to another row."""
    return m[m.duplicated("file_sha256", keep=False)].sort_values("file_sha256")


def resolution_summary(m):
    w, h = m["downloaded_width"], m["downloaded_height"]
    short, long_ = pd.concat([w, h], axis=1).min(axis=1), pd.concat([w, h], axis=1).max(axis=1)
    out = m[["photo_id", "class_id", "common_name", "split", "local_path"]].copy()
    out["width"], out["height"] = w, h
    out["short_side"], out["aspect"] = short, long_ / short
    out["too_small"] = short < SMALL_SIDE_PX
    out["extreme_aspect"] = out["aspect"] > EXTREME_ASPECT
    return out


def downloaded_training_rows(train, root=REPO_ROOT):
    """Find local training files and reject paths that leave the training split."""
    root = Path(root)
    image_dir = Path("data/raw/images")
    relative_train_dir = image_dir / "train"
    train_dir = (root / image_dir).resolve() / "train"
    found = []
    for value in train["local_path"]:
        relative = Path(value)
        path = (root / relative).resolve()
        if not relative.is_relative_to(relative_train_dir) or not path.is_relative_to(train_dir):
            raise ValueError(f"Training image path leaves the training split: {relative}")
        found.append(path.is_file())
    return train.loc[found]


def inspect_images(rows, root=REPO_ROOT):
    """Open each image and record properties that matter for training."""
    from PIL import Image, ImageStat
    records = []
    for row in rows.itertuples():
        rec = {"local_path": row.local_path, "common_name": row.common_name,
               "readable": True, "mode": None, "dims_match_manifest": None,
               "exif_rotated": None, "mean_brightness": None, "error": ""}
        try:
            with Image.open(Path(root) / row.local_path) as im:
                im.load()
                rec["mode"] = im.mode
                width, height = row.downloaded_width, row.downloaded_height
                if pd.notna(width) and pd.notna(height):
                    rec["dims_match_manifest"] = im.size == (width, height)
                rec["exif_rotated"] = im.getexif().get(0x0112, 1) not in (1, None)
                rec["mean_brightness"] = round(ImageStat.Stat(im.convert("L")).mean[0], 1)
        except Exception as e:
            rec["readable"], rec["error"] = False, str(e)
        records.append(rec)
    return pd.DataFrame.from_records(records, columns=[
        "local_path", "common_name", "readable", "mode", "dims_match_manifest",
        "exif_rotated", "mean_brightness", "error",
    ])


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def make_species_grid(rows, out_path, per_class=6, thumb=160, root=REPO_ROOT):
    """One row of thumbnails per species, labelled, saved as a single PNG."""
    from PIL import Image, ImageDraw, ImageOps
    label_w = 220
    groups = list(rows.groupby(["class_id", "common_name"]))
    sheet = Image.new("RGB", (label_w + per_class * thumb, len(groups) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for r, ((_, name), g) in enumerate(groups):
        draw.text((8, r * thumb + thumb // 2 - 6), name, fill="black")
        for c, path in enumerate(g["local_path"].head(per_class)):
            img = ImageOps.fit(_open_rgb(Path(root) / path), (thumb, thumb))
            sheet.paste(img, (label_w + c * thumb, r * thumb))
    sheet.save(out_path)
    return len(groups)


def _augmentation_panels(im, seed=AUGMENTATION_SEED):
    """Build PIL previews of the evaluation geometry and candidate transforms."""
    import torch
    from torchvision import transforms
    from torchvision.models import EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    evaluate = transforms.Compose([
        transforms.Resize(weights.resize_size, interpolation=weights.interpolation,
                          antialias=weights.antialias),
        transforms.CenterCrop(weights.crop_size),
    ])
    crop = transforms.RandomResizedCrop(weights.crop_size, scale=(0.6, 1.0),
                                        interpolation=weights.interpolation)
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    train = transforms.Compose([crop, transforms.RandomHorizontalFlip(p=0.5), jitter])
    base = evaluate(im)

    # The PIL transforms sample from the CPU generator. Preserve the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return [
            ("Original", im),
            ("Resize + center crop\n(eval)", base),
            ("RandomResizedCrop\nscale 0.6–1.0", crop(im)),
            ("Horizontal flip\n(forced)", transforms.RandomHorizontalFlip(p=1.0)(base)),
            ("ColorJitter 0.3", jitter(base)),
            ("Rotation ±15°\n(omit by default)", transforms.RandomRotation(15)(base)),
            ("Vertical flip\n(not recommended)", transforms.RandomVerticalFlip(p=1.0)(base)),
            ("Recommended training\ncrop → flip → jitter", train(im)),
        ]


def make_augmentation_examples(rows, out_dir, root=REPO_ROOT):
    """For one image per animal group, save a before/after strip of augmentations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = rows.sort_values(["class_id", "photo_id"]).groupby("group").head(1)
    saved = []
    for row in picks.itertuples():
        im = _open_rgb(Path(root) / row.local_path)
        panels = _augmentation_panels(im)
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
        for ax, (title, preview) in zip(axes, panels):
            ax.imshow(preview)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{row.common_name} ({row.group})")
        plt.tight_layout()
        path = out_dir / f"augment_{row.common_name.replace(' ', '_')}.png"
        plt.savefig(path, dpi=110)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_class_distribution(dist, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = [c for c in dist.columns if c != "total"]
    names = [name for _, name in dist.index]
    ax = dist[splits].set_axis(names).plot(kind="barh", stacked=True, figsize=(10, 9))
    ax.invert_yaxis()
    ax.set_xlabel("Number of photos")
    ax.set_title("WildQuest photos per species and split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--class-map", default=DEFAULT_CLASS_MAP)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--images", action="store_true",
                   help="inspect training images only and save species and augmentation figures")
    p.add_argument("--root", default=REPO_ROOT, help="folder that local_path is relative to")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(args.manifest)
    lines = [f"Rows: {len(m)}   Species: {m['class_id'].nunique()}   "
             f"Observations: {m['observation_uuid'].nunique()}"]

    cmap = load_class_mapping(args.class_map)
    if cmap is not None:
        label_to_id = cmap.get("label_to_id", cmap)
        lines.append(f"class_mapping.json species: {len(label_to_id)}")
        pairs = m[["scientific_name", "class_id"]].drop_duplicates()
        mismatched = [row.scientific_name for row in pairs.itertuples()
                      if label_to_id.get(row.scientific_name) != row.class_id]
        lines.append("Manifest species whose class_id disagrees with "
                     f"class_mapping.json: {len(mismatched)}")

    dist = class_distribution(m)
    dist.to_csv(out / "class_distribution.csv")
    observation_distribution(m).to_csv(out / "observations_per_class.csv")
    plot_class_distribution(dist, out / "class_distribution.png")
    lines += ["", "Photos per split:", m["split"].value_counts().to_string(),
              "", "Photos per species (all splits):",
              f"  min {dist['total'].min()}, max {dist['total'].max()}, "
              f"ratio {dist['total'].max() / dist['total'].min():.2f}",
              "", "Photos per group:", m["group"].value_counts().to_string(),
              "", "Licenses:", m["license"].value_counts().to_string(),
              "", "DMV-located photos per split:",
              m.groupby("split")["is_dmv_bbox"].mean().round(3).to_string(),
              "", "Photos per observation:", photos_per_observation(m).round(2).to_string()]

    leaked = leakage_check(m)
    lines += ["", f"Observations found in more than one split: {len(leaked)}"]

    dups = duplicate_files(m)
    dups.to_csv(out / "duplicate_files.csv", index=False)
    lines.append(f"Rows with byte-identical duplicate images: {len(dups)}")

    res = resolution_summary(m)
    res.to_csv(out / "resolution.csv", index=False)
    lines += ["", "Short side (px):", res["short_side"].describe().round(1).to_string(),
              f"Images with short side < {SMALL_SIDE_PX}px: {int(res['too_small'].sum())}",
              f"Images with aspect ratio > {EXTREME_ASPECT}: {int(res['extreme_aspect'].sum())}"]

    if args.images:
        train = m.loc[m["split"].eq("train")]
        rows = downloaded_training_rows(train, args.root)
        info = inspect_images(rows, args.root)
        info.to_csv(out / "image_inspection.csv", index=False)
        ok = info.loc[info["readable"].eq(True)]
        modes = ", ".join(f"{mode}={count}" for mode, count in ok["mode"].value_counts().items())
        lines += ["", f"Eligible training images found locally: {len(rows)} of {len(train)}",
                  f"Missing training images: {len(train) - len(rows)}",
                  f"Training images inspected: {len(info)} of {len(train)}",
                  f"Unreadable images: {int(info['readable'].eq(False).sum())}",
                  f"Color modes: {modes or 'none'}",
                  f"Size differs from manifest: {int(ok['dims_match_manifest'].eq(False).sum())}",
                  f"Size unavailable in manifest: {int(ok['dims_match_manifest'].isna().sum())}",
                  f"EXIF-rotated images: {int(ok['exif_rotated'].eq(True).sum())}",
                  f"Very dark images (mean brightness < 40): "
                  f"{int((ok['mean_brightness'] < 40).sum())}",
                  f"Very bright images (mean brightness > 215): "
                  f"{int((ok['mean_brightness'] > 215).sum())}"]
        if ok.empty:
            lines.append("Figures skipped: no readable training images.")
        else:
            good = rows[rows["local_path"].isin(ok["local_path"])]
            n = make_species_grid(good, out / "species_grid.png", root=args.root)
            lines.append(f"Species grid saved with {n} species")
            saved = make_augmentation_examples(good, out / "augmentations", root=args.root)
            lines.append(f"Augmentation examples saved: {len(saved)}")

    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
