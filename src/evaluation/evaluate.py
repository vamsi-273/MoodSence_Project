import torch
import numpy as np  # noqa: F401
from sklearn.metrics import classification_report, confusion_matrix
from src.models.mobilenet_model import get_mobilenet_model
from src.training.dataloader import get_dataloaders


def evaluate_model(model_path="best_model.pth"):

    device = torch.device("cpu")

    model = get_mobilenet_model(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, val_loader = get_dataloaders(batch_size=16)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))