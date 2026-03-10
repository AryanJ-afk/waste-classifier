from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from src.data.dataset import get_dataloaders
from src.models.cnn import SimpleCNN


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5
MODEL_SAVE_PATH = Path("saved_models/baseline_cnn.pth")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, _, class_names = get_dataloaders()
    print("Classes:", class_names)

    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)
        
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()