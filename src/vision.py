"""Image reading, colour labels, and the two encoders used by the matcher.

CLIP is the pretrained shared embedding from the proposal (no fine-tuning).
The CNN is a frozen MobileNetV3 backbone; only a linear softmax head is trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

COLOURS = [
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "pink",
]

CATEGORIES = ["key", "umbrella", "bottle"]

DISPLAY_CATEGORY = {
    "key": "key",
    "umbrella": "umbrella",
    "bottle": "water bottle",
}

# Reference colours in RGB. Chromatic names are used only when the photo is saturated.
PALETTE = {
    "black": np.array([28, 28, 28], dtype=np.float32),
    "white": np.array([236, 236, 236], dtype=np.float32),
    "gray": np.array([148, 148, 148], dtype=np.float32),
    "red": np.array([186, 42, 48], dtype=np.float32),
    "orange": np.array([214, 118, 38], dtype=np.float32),
    "yellow": np.array([224, 190, 48], dtype=np.float32),
    "green": np.array([46, 140, 72], dtype=np.float32),
    "blue": np.array([46, 92, 184], dtype=np.float32),
    "purple": np.array([124, 58, 164], dtype=np.float32),
    "brown": np.array([122, 78, 42], dtype=np.float32),
    "pink": np.array([220, 136, 164], dtype=np.float32),
}
CHROMATIC = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "pink"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=lambda path: path.as_posix().lower())
    return files


def can_read(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except Exception:
        return False


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def read_rgb_array(path: Path, size: int = 96) -> np.ndarray:
    image = open_rgb(path)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image)


def save_thumb(path: Path, dest: Path, size: int = 420) -> None:
    image = open_rgb(path)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "JPEG", quality=86)


def _hue_sat_val(pix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red, green, blue = pix[:, :, 0], pix[:, :, 1], pix[:, :, 2]
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    delta = max_channel - min_channel
    hue = np.zeros_like(max_channel)
    nonzero = delta > 1e-6
    red_mask = nonzero & (max_channel == red)
    green_mask = nonzero & (max_channel == green)
    blue_mask = nonzero & (max_channel == blue)
    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4
    hue = hue / 6.0
    saturation = np.zeros_like(max_channel)
    positive = max_channel > 1e-6
    saturation[positive] = delta[positive] / max_channel[positive]
    return hue, saturation, max_channel


def _neutral_name(value: np.ndarray) -> str:
    if float(np.median(value)) < 0.32 or float(np.mean(value < 0.28)) > 0.45:
        return "black"
    if float(np.median(value)) > 0.82:
        return "white"
    return "gray"


def _hue_name(hue: float, value: float) -> str:
    """Map a hue in [0, 1] and a brightness in [0, 1] onto the fixed colour list."""
    if hue < 0.03 or hue >= 0.96:
        if value > 0.72:
            return "pink"
        if value < 0.42:
            return "brown"
        return "red"
    if hue < 0.09:
        if value < 0.45:
            return "brown"
        return "orange"
    if hue < 0.17:
        if value < 0.45:
            return "brown"
        return "yellow"
    if hue < 0.46:
        return "green"
    if hue < 0.75:
        return "blue"
    if hue < 0.86:
        return "purple"
    if value > 0.65:
        return "pink"
    return "purple"


def dominant_colour(rgb: np.ndarray) -> str:
    """Name the object colour. The centre of the photo counts more than the border."""
    height, width, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    frame = np.concatenate(
        [
            rgb[:6, :, :].reshape(-1, 3),
            rgb[-6:, :, :].reshape(-1, 3),
            rgb[:, :6, :].reshape(-1, 3),
            rgb[:, -6:, :].reshape(-1, 3),
        ]
    ).astype(np.float32)
    border = np.median(frame, axis=0)
    distance = np.linalg.norm(pixels - border, axis=1)
    max_channel = pixels.max(axis=1)
    min_channel = pixels.min(axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0)
    value = max_channel / 255.0
    yy, xx = np.mgrid[0:height, 0:width]
    centre_weight = np.exp(
        -(((yy - (height - 1) / 2) / (height * 0.28)) ** 2 + ((xx - (width - 1) / 2) / (width * 0.28)) ** 2)
    ).reshape(-1).astype(np.float32)
    foreground = distance > 28
    foreground &= ~((value > 0.94) & (saturation < 0.10))
    if float((foreground * centre_weight).sum()) < 0.04 * float(centre_weight.sum()):
        foreground = ~((value > 0.92) & (saturation < 0.14))
    if not np.any(foreground):
        foreground = np.ones(len(pixels), dtype=bool)

    chosen = pixels[foreground] / 255.0
    chosen_sat = saturation[foreground]
    chosen_val = value[foreground]
    chosen_weight = centre_weight[foreground]
    strong = chosen_sat > 0.42
    neutral_mass = float(chosen_weight[~strong].sum()) if np.any(~strong) else 0.0
    if int(strong.sum()) >= 20:
        hue, _, _ = _hue_sat_val(chosen.reshape(-1, 1, 3))
        hue = hue.reshape(-1)
        votes: dict[str, float] = {}
        for hue_value, sat_value, pixel_value, weight in zip(
            hue[strong], chosen_sat[strong], chosen_val[strong], chosen_weight[strong]
        ):
            name = _hue_name(float(hue_value), float(pixel_value))
            votes[name] = votes.get(name, 0.0) + float(weight) * float(sat_value)
        best_name = max(votes, key=votes.get)
        if votes[best_name] >= 0.35 * max(neutral_mass, 1e-6):
            return best_name
    return _neutral_name(chosen_val[np.argsort(-chosen_weight)[: max(1, len(chosen_val) // 3)]])


def _block_mean(gray: np.ndarray, blocks: int) -> np.ndarray:
    height, width = gray.shape
    block_h, block_w = height // blocks, width // blocks
    cropped = gray[: block_h * blocks, : block_w * blocks]
    return cropped.reshape(blocks, block_h, blocks, block_w).mean(axis=(1, 3))


def handcrafted_features(rgb: np.ndarray) -> np.ndarray:
    """Fixed visual features used if the pretrained networks cannot be loaded."""
    pixels = rgb.astype(np.float32) / 255.0
    gray = pixels.mean(axis=2)
    hue, saturation, value = _hue_sat_val(pixels)
    chromatic = saturation > 0.15
    if np.any(chromatic):
        hist, _ = np.histogram(hue[chromatic], bins=12, range=(0.0, 1.0))
    else:
        hist = np.zeros(12, dtype=np.float64)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    gradient_x = float(np.abs(np.diff(gray, axis=1)).mean())
    gradient_y = float(np.abs(np.diff(gray, axis=0)).mean())
    parts = [
        _block_mean(gray, 8).reshape(-1).astype(np.float32),
        pixels.mean(axis=(0, 1)).astype(np.float32),
        pixels.std(axis=(0, 1)).astype(np.float32),
        hist,
        np.array(
            [float(saturation.mean()), float(value.mean()), gradient_x, gradient_y],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(parts)


def text_prompt(colour: str, category: str) -> str:
    return f"a photo of a {colour} {DISPLAY_CATEGORY[category]}"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)


class ClipEncoder:
    """Pretrained CLIP ViT-B/32. The proposal uses it without fine-tuning."""

    def __init__(self) -> None:
        import open_clip
        import torch

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        last_error: Exception | None = None
        model = preprocess = None
        chosen = None
        for pretrained in ("openai", "laion2b_s34b_b79k"):
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained=pretrained
                )
                chosen = pretrained
                break
            except Exception as exc:  # network or weight-name changes
                last_error = exc
                print(f"CLIP weights '{pretrained}' unavailable: {exc}", flush=True)
        if model is None or preprocess is None or chosen is None:
            raise RuntimeError(f"Could not load CLIP ViT-B/32 ({last_error})")
        model.eval()
        self.torch = torch
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.pretrained = chosen
        self.name = f"ViT-B/32 ({chosen})"
        print(f"Loaded CLIP {self.name}", flush=True)

    def encode_images(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        outputs = []
        total = len(paths)
        for start in range(0, total, batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [self.preprocess(open_rgb(path)) for path in batch_paths]
            batch = self.torch.stack(images)
            with self.torch.inference_mode():
                encoded = self.model.encode_image(batch)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            outputs.append(encoded.cpu().numpy().astype(np.float32))
            print(f"  CLIP images {min(start + batch_size, total)}/{total}", flush=True)
        return np.concatenate(outputs, axis=0)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts)
        with self.torch.inference_mode():
            encoded = self.model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded.cpu().numpy().astype(np.float32)


class CnnEncoder:
    """Frozen MobileNetV3-Small. The trainable part is a separate linear softmax."""

    def __init__(self) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        self.torch = torch
        self.model = model
        self.preprocess = weights.transforms()
        self.name = "MobileNetV3-Small (ImageNet, frozen)"
        print(f"Loaded {self.name}", flush=True)

    def encode_images(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        outputs = []
        total = len(paths)
        for start in range(0, total, batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [self.preprocess(open_rgb(path)) for path in batch_paths]
            batch = self.torch.stack(images)
            with self.torch.inference_mode():
                encoded = self.model(batch)
            outputs.append(encoded.cpu().numpy().astype(np.float32))
            print(f"  CNN features {min(start + batch_size, total)}/{total}", flush=True)
        return np.concatenate(outputs, axis=0)
