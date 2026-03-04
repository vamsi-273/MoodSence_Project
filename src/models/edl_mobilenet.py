import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from src.models.se_block import SEBlock


class EDL_MobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EDL_MobileNet, self).__init__()

        self.num_classes = num_classes

        self.backbone = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 3 blocks
        for param in self.backbone.features[-5:].parameters():
            param.requires_grad = True

        self.se = SEBlock(576)

        # 🔥 IMPORTANT: feature size = 576
        self.evidence_layer = nn.Linear(576, num_classes)

    def forward(self, x):

        x = self.backbone.features(x)
        x = self.se(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        evidence = F.softplus(self.evidence_layer(x))

        return evidence