from src.training.dataloader import get_dataloaders
import torch


if __name__ == "__main__":

    train_loader, _ = get_dataloaders(batch_size=1)

    class_counts = [0] * 7

    for _, labels in train_loader:
        class_counts[labels.item()] += 1

    print("Class counts:", class_counts)

    total = sum(class_counts)
    weights = [total / c for c in class_counts]

    print("Raw weights:", weights)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    print("Normalized weights:", weights)