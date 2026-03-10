from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.models.transformer import get_vit_model


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = Path("saved_models/baseline_transformer.pth")


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

    loss = running_loss / total
    acc = correct / total
    return loss, acc


def main():
    _, _, test_loader, class_names = get_dataloaders(batch_size=8)

    model = get_vit_model(
        num_classes=len(class_names),
        freeze_backbone=True
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Using device: {DEVICE}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()