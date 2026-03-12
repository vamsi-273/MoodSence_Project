import torch
# import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.models.edl_mobilenet import EDL_MobileNet
from src.training.dataloader import get_dataloaders


def evaluate_edl(model_path="best_edl_model.pth"):

    device = torch.device("cpu")

    model = EDL_MobileNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, val_loader = get_dataloaders(batch_size=16)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            evidence = model(images)
            alpha = evidence + 1
            probs = alpha / torch.sum(alpha, dim=1, keepdim=True)

            _, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # print("\nAccuracy:")
    # print(accuracy_score(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))