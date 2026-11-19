"""Small CPU-only tests. Run: python -m unittest discover -s Code/tests -v"""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wildquest.classifier.dataset import WildlifeDataset
from wildquest.classifier.model import build_model, set_trainable_feature_blocks, make_optimizer


class ClassifierTests(unittest.TestCase):
    def test_dataset_reads_frozen_ids_and_excludes_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "Code/configs"
            config.mkdir(parents=True)
            labels = {str(i): f"species_{i}" for i in range(25)}
            (config / "class_mapping.json").write_text(json.dumps({"id_to_label": labels}))
            fields = ("class_id", "split", "local_path", "scientific_name")
            with (config / "dataset_manifest.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for split in ("train", "val", "test", "dmv_test"):
                    relative = f"data/raw/images/{split}/03/example.jpg"
                    path = root / relative
                    path.parent.mkdir(parents=True)
                    Image.new("RGB", (300, 250), color="orange").save(path)
                    writer.writerow(dict(class_id="3", split=split, local_path=relative,
                                         scientific_name="species_3"))
            train = WildlifeDataset(root, "train")
            val = WildlifeDataset(root, "val")
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)
            image, label = train[0]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(label, 3)
            with self.assertRaises(ValueError):
                WildlifeDataset(root, "test")
            with self.assertRaises(ValueError):
                WildlifeDataset(root, "dmv_test")

    def test_model_output_and_freezing(self):
        model = build_model(pretrained=False)
        self.assertEqual(model.classifier[1].out_features, 25)
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        optimizer_head = make_optimizer(model)
        self.assertEqual(len(optimizer_head.param_groups), 1)

        set_trainable_feature_blocks(model, 2)
        self.assertFalse(any(p.requires_grad for p in model.features[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-1].parameters()))
        optimizer_finetune = make_optimizer(model)
        self.assertEqual(len(optimizer_finetune.param_groups), 2)
        self.assertLess(optimizer_finetune.param_groups[1]["lr"],
                        optimizer_finetune.param_groups[0]["lr"])

    def test_single_training_and_validation_step_cpu(self):
        # Tiny synthetic model tests gradient flow without downloading ImageNet weights.
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader
        script = Path(__file__).resolve().parents[1] / "scripts/train_classifier.py"
        spec = importlib.util.spec_from_file_location("train_classifier", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.randn(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 3])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Flatten())
                self.classifier = nn.Sequential(nn.Linear(3 * 8 * 8, 25))

            def forward(self, x):
                return self.classifier(self.features(x))

        model = TinyModel()
        before = model.classifier[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss, accuracy = module.run_epoch(model, loader, torch.device("cpu"),
                                          nn.CrossEntropyLoss(), optimizer, scaler)
        self.assertGreater(loss, 0)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertFalse(torch.equal(before, model.classifier[0].weight))
        val_loss, val_acc = module.run_epoch(model, loader, torch.device("cpu"),
                                             nn.CrossEntropyLoss())
        self.assertGreater(val_loss, 0)
        self.assertTrue(0 <= val_acc <= 1)

    def test_checkpoint_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            original = build_model(pretrained=False)
            torch.save({"model_state_dict": original.state_dict()}, path)
            restored = build_model(pretrained=False)
            data = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(data["model_state_dict"])
            self.assertTrue(torch.equal(original.classifier[1].weight,
                                        restored.classifier[1].weight))

    def test_invalid_unfreeze_depth(self):
        model = build_model(pretrained=False)
        with self.assertRaises(ValueError):
            set_trainable_feature_blocks(model, -1)
        with self.assertRaises(ValueError):
            set_trainable_feature_blocks(model, 100)


if __name__ == "__main__":
    unittest.main()
