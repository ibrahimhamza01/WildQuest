"""Audit the frozen WildQuest dataset (issue #4).

Reads Code/configs/dataset_manifest.csv (read-only) and writes summaries to
outputs/dataset_audit/. Never modifies the manifest or splits.

Usage:
    python3 Code/scripts/audit_dataset.py
    python3 Code/scripts/audit_dataset.py --check-images   # also open each image file
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
    out["short_side"], out["aspect"] = short, (long_ / short).round(2)
    out["too_small"] = short < SMALL_SIDE_PX
    out["extreme_aspect"] = out["aspect"] > EXTREME_ASPECT
    return out


def check_image_files(m, root=REPO_ROOT):
    """Try to open every image; return rows that are missing or unreadable."""
    from PIL import Image
    problems = []
    for path in m["local_path"]:
        full = Path(root) / path
        if not full.exists():
            problems.append((path, "missing"))
            continue
        try:
            with Image.open(full) as im:
                im.verify()
        except Exception as e:
            problems.append((path, f"unreadable: {e}"))
    return pd.DataFrame(problems, columns=["local_path", "problem"])


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
    p.add_argument("--check-images", action="store_true", help="open every image file")
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

    if args.check_images:
        bad = check_image_files(m)
        bad.to_csv(out / "image_problems.csv", index=False)
        lines.append(f"Missing or unreadable image files: {len(bad)}")

    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()