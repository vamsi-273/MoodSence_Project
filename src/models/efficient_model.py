import torch.nn as nn
from torchvision import models


def get_efficientnet_model(num_classes=7):

    model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model