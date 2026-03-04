import torch
import numpy as np
from src.models.edl_mobilenet import EDL_MobileNet
from src.training.dataloader import get_dataloaders


def evaluate_edl_uncertainty(model_path="best_edl_model.pth"):

    device = torch.device("cpu")

    model = EDL_MobileNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, val_loader = get_dataloaders(batch_size=16)

    correct_uncertainties = []
    wrong_uncertainties = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            evidence = model(images)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)

            probs = alpha / S
            uncertainty = 7 / S

            _, predicted = torch.max(probs, 1)

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_uncertainties.append(uncertainty[i].item())
                else:
                    wrong_uncertainties.append(uncertainty[i].item())

    print("Average uncertainty (Correct):",
          np.mean(correct_uncertainties))

    print("Average uncertainty (Wrong):",
          np.mean(wrong_uncertainties))