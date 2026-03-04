import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.mobilenet_model import get_mobilenet_model
from src.training.dataloader import get_dataloaders


def train_model(epochs=10, batch_size=16, lr=0.001):

    device = torch.device("cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = get_mobilenet_model(num_classes=7)
    model.to(device)

    class_weights = torch.tensor(
        [0.0939, 0.4310, 0.1689, 0.0254, 0.0611, 0.1718, 0.0480],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0003,
        weight_decay=1e-4
    )

    best_val_acc = 0.0

    for epoch in range(epochs):

        # ---------- TRAINING ----------
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # ---------- VALIDATION ----------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    print("Training complete.")