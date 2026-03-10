import torch.nn as nn
from torchvision import models


def get_vit_model(num_classes: int = 6, freeze_backbone: bool = True):
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model