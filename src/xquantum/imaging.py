"""Bounded image decoding and a transparent CPU baseline feature extractor."""

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
FEATURE_VERSION = "rgb-gradient-grid-v1"


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Image must contain 1 to 10485760 bytes")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Only JPEG, PNG and WEBP are supported")
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 20 million pixels")
                if min(original.size) < 64:
                    raise ValueError("Image dimensions must both be at least 64 pixels")
                return ImageOps.exif_transpose(original).convert("RGB").copy()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid or unsafe image") from exc


def quality_issues(image: Image.Image) -> list[str]:
    issues = []
    if min(image.size) < 128:
        issues.append("low_resolution")
    gray = np.asarray(image.resize((128, 128)).convert("L"), dtype=np.float64)
    if gray.std() < 5:
        issues.append("low_contrast")
    mean = gray.mean()
    if mean < 15 or mean > 240:
        issues.append("extreme_exposure")
    return issues


def perceptual_hash(image: Image.Image) -> int:
    gray = np.asarray(image.resize((9, 8)).convert("L"))
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return sum(int(bit) << index for index, bit in enumerate(bits))


def image_features(image: Image.Image) -> np.ndarray:
    """432 color/gradient statistics; no anatomical segmentation is implied."""
    pixels = np.asarray(ImageOps.pad(image, (128, 128)), dtype=np.float64) / 255
    gray = pixels.mean(axis=2)
    dy, dx = np.gradient(gray)
    angles = (np.arctan2(dy, dx) + np.pi) % np.pi
    magnitudes = np.hypot(dx, dy)
    features = []
    for y in range(0, 128, 32):
        for x in range(0, 128, 32):
            cell = pixels[y : y + 32, x : x + 32]
            for channel in range(3):
                hist, _ = np.histogram(cell[:, :, channel], bins=6, range=(0, 1))
                features.extend(hist / cell[:, :, channel].size)
            hist, _ = np.histogram(
                angles[y : y + 32, x : x + 32],
                bins=9,
                range=(0, np.pi),
                weights=magnitudes[y : y + 32, x : x + 32],
            )
            features.extend(hist / (np.linalg.norm(hist) + 1e-8))
    return np.asarray(features, dtype=np.float64)
