from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from src.models.cnn import SimpleCNN
from src.models.transformer import get_vit_model


DEVICE = torch.device("cpu")

CNN_MODEL_PATH = Path("saved_models/baseline_cnn.pth")
TRANSFORMER_MODEL_PATH = Path("saved_models/baseline_transformer.pth")

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_loaded_models = {}


def load_cnn_model():
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_transformer_model():
    model = get_vit_model(num_classes=len(CLASS_NAMES), freeze_backbone=True).to(DEVICE)
    model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def get_model(model_type: str):
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    if model_type == "cnn":
        model = load_cnn_model()
    elif model_type == "transformer":
        model = load_transformer_model()
    else:
        raise ValueError("model_type must be either 'cnn' or 'transformer'")

    _loaded_models[model_type] = model
    return model


def predict_image(image: Image.Image, model_type: str):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = get_model(model_type)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return {
        "predicted_class": CLASS_NAMES[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
        "model_used": model_type
    }