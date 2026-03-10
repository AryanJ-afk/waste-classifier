import torch
from src.models.transformer import get_vit_model


def main():
    model = get_vit_model(num_classes=6)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)

    print(type(model).__name__)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()