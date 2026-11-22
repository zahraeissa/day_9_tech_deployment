from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        archive_buffer = BytesIO()
        with ZipFile(archive_buffer, "w", compression=ZIP_STORED) as archive:
            for path in checkpoint_path.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(checkpoint_path.parent))
        archive_buffer.seek(0)
        return torch.load(archive_buffer, map_location=device, weights_only=True)

    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_inference_transform(checkpoint: dict) -> transforms.Compose:
    normalization = checkpoint["normalization"]
    return transforms.Compose(
        [
            transforms.Resize(checkpoint["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                normalization["mean"],
                normalization["std"],
            ),
        ]
    )


def preprocess_image(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        probabilities = model(image_tensor.to(device)).softmax(dim=1)[0]
    
    scores, indices = probabilities.topk(min(top_k, len(class_names)))
    return [
        (class_names[index.item()], score.item())
        for score, index in zip(scores, indices)
    ]

