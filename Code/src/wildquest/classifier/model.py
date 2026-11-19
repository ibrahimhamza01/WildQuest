"""EfficientNet-B0 model construction and two-stage transfer learning."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def set_trainable_feature_blocks(model, last_n):
    """Train the last N feature blocks; always train the new 25-class head."""
    blocks = list(model.features.children())
    if not 0 <= last_n <= len(blocks):
        raise ValueError(f"last_n must be between 0 and {len(blocks)}")
    for i, block in enumerate(blocks):
        for parameter in block.parameters():
            parameter.requires_grad = i >= len(blocks) - last_n
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def eval_frozen_batchnorm(model):
    """Keep running statistics fixed in feature blocks whose parameters are frozen.

    Call just after model.train(), which otherwise switches every BatchNorm to train mode.
    """
    for block in model.features:
        if not any(parameter.requires_grad for parameter in block.parameters()):
            for module in block.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()


def build_model(num_classes=25, pretrained=True, last_unfrozen_blocks=0):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    set_trainable_feature_blocks(model, last_unfrozen_blocks)
    return model


def make_optimizer(model, head_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4):
    """Use a smaller learning rate for pretrained features than for the new head."""
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": head_lr}]
    if feature_params:
        groups.append({"params": feature_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
