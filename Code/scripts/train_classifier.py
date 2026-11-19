#!/usr/bin/env python3
"""Two-stage EfficientNet-B0 training using ONLY the frozen train/val splits."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.classifier.dataset import make_dataloaders
from wildquest.classifier.model import (
    build_model, eval_frozen_batchnorm, make_optimizer, set_trainable_feature_blocks,
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None):
    """Returns sample-weighted (loss, accuracy). optimizer=None means validation."""
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        eval_frozen_batchnorm(model)
    # Match Python float accumulation for the loss used to break checkpoint ties.
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        n = labels.size(0)
        total_loss.add_(loss.detach(), alpha=n)
        total_correct.add_((logits.argmax(dim=1) == labels).sum())
        total += n
    if not total:
        raise ValueError("Empty DataLoader")
    return total_loss.item() / total, total_correct.item() / total


def train(config, smoke_test=False, device_name="cuda"):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU not found; use --device cpu for local debugging")
    device = torch.device(device_name)
    seed_everything(config["seed"])
    output = ROOT / config["output_dir"]
    if smoke_test:
        output = output.with_name(output.name + "_smoke")
    output.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        ROOT, batch_size=4 if smoke_test else config["batch_size"],
        num_workers=0 if smoke_test else config["num_workers"], seed=config["seed"],
        pin_memory=device.type == "cuda", persistent_workers=config.get("persistent_workers", False),
    )
    if smoke_test:
        # Deterministic, tiny subsets: verify mechanics, not model quality.
        generator = torch.Generator().manual_seed(config["seed"])
        def small_loader(loader):
            count = min(len(loader.dataset), 8)
            indices = torch.randperm(len(loader.dataset), generator=generator)[:count].tolist()
            return DataLoader(Subset(loader.dataset, indices), batch_size=4, shuffle=False,
                              pin_memory=loader.pin_memory)
        train_loader = small_loader(train_loader)
        val_loader = small_loader(val_loader)

    mapping = json.loads((ROOT / "Code/configs/class_mapping.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256((ROOT / "Code/configs/dataset_manifest.csv").read_bytes()).hexdigest()
    model = build_model(num_classes=25, pretrained=True, last_unfrozen_blocks=0).to(device)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_acc = -1.0
    best_loss = float("inf")
    patience = config["early_stopping_patience"]
    log_file = output / "history.csv"
    rows = []
    epoch_number = 0

    stages = [("head", 1 if smoke_test else config["head_epochs"], 0)]
    if not smoke_test:
        stages.append(("fine_tune", config["finetune_epochs"], config["unfreeze_last_n"]))

    for stage, epochs, unfrozen in stages:
        set_trainable_feature_blocks(model, unfrozen)
        optimizer = make_optimizer(
            model, head_lr=config["head_lr"], backbone_lr=config["backbone_lr"],
            weight_decay=config["weight_decay"],
        )
        stale = 0
        for _ in range(epochs):
            epoch_number += 1
            start = perf_counter()
            train_loss, train_acc = run_epoch(model, train_loader, device, loss_fn, optimizer, scaler)
            train_seconds = perf_counter() - start
            val_loss, val_acc = run_epoch(model, val_loader, device, loss_fn)
            epoch_seconds = perf_counter() - start
            row = dict(epoch=epoch_number, stage=stage, train_loss=train_loss,
                       train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_acc)
            rows.append(row)
            with log_file.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"epoch={epoch_number} stage={stage} train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"train_images/s={len(train_loader.dataset) / train_seconds:.1f} "
                  f"train_s={train_seconds:.2f} train_val_s={epoch_seconds:.2f}", flush=True)
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc, best_loss, stale = val_acc, val_loss, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch_number,
                    "stage": stage,
                    "last_unfrozen_blocks": unfrozen,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "class_mapping": mapping,
                    "config": config,
                    "manifest_sha256": manifest_hash,
                    "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
                }, output / "best.pt")
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping stage={stage} after {stale} non-improving epochs")
                    break
    print(f"Best validation accuracy: {best_acc:.4f}; files: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Code/configs/train_efficientnet_b0.json")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train(config, smoke_test=args.smoke_test, device_name=args.device)


if __name__ == "__main__":
    main()
