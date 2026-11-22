from pathlib import Path
import torch
from PIL import Image

from utils import (
    build_inference_transform,
    build_model,
    load_checkpoint,
    predict,
    preprocess_image,
)


CHECKPOINT = Path(__file__).resolve().parent / "resnet18_cifar10_deploy"
def get_device():
    """Utility function to select the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()


def load_model_for_inference(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(checkpoint, device)
    transform = build_inference_transform(checkpoint)
    return model, checkpoint["class_names"], transform


def main() -> None:
    model, class_names, transform = load_model_for_inference(
        CHECKPOINT,
        DEVICE,
    )