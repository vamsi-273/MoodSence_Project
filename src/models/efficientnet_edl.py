# import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EfficientNetEDL(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        in_features = self.backbone.classifier[1].in_features

        # remove classifier
        self.backbone.classifier = nn.Identity()

        self.evidence_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):

        features = self.backbone(x)

        evidence = F.softplus(self.evidence_layer(features))

        return evidence