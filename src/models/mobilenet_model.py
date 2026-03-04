import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from src.models.se_block import SEBlock


class MobileNetWithSE(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNetWithSE, self).__init__()

        self.backbone = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 3 feature blocks
        for param in self.backbone.features[-3:].parameters():
            param.requires_grad = True

        self.se = SEBlock(576)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


def get_mobilenet_model(num_classes=7):
    return MobileNetWithSE(num_classes)