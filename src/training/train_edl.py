import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.edl_mobilenet import EDL_MobileNet
from src.models.edl_loss import edl_loss
from src.training.dataloader import get_dataloaders


def train_edl_model(epochs=15, batch_size=16, lr=0.0005):

    device = torch.device("cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = EDL_MobileNet(num_classes=7).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            evidence = model(images)

            loss = edl_loss(
                evidence,
                labels,
                num_classes=7,
                epoch=epoch,
                annealing_step=50
            )

            loss.backward()
            optimizer.step()

            alpha = evidence + 1
            probs = alpha / torch.sum(alpha, dim=1, keepdim=True)

            _, predicted = torch.max(probs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        print(f"\nEpoch {epoch} | Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                evidence = model(images)
                alpha = evidence + 1
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)

                _, predicted = torch.max(probs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Validation Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_edl_model.pth")
            print("EDL Model saved!")

    print("Training complete.")