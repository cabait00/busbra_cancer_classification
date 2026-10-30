"""Deterministic image preparation for the BUS-BRA classification project.

The functions turn one original ultrasound PNG into a fixed-size grayscale array.
Only pixels are used. No mask, bounding box, metadata or learned statistic enters.

Geometry (the contract of resize_with_padding)
----------------------------------------------
1. The WHOLE image is scaled down proportionally so that it fits into the target
   square/rectangle (PIL.ImageOps.contain with bilinear interpolation). The aspect
   ratio is kept; nothing is cropped and nothing is stretched.
2. Pillow rounds the scaled side with Python's round(), i.e. half to even.
3. The scaled image is centred on a black canvas (pixel value 0). If the free space
   is odd, the extra pixel lies on the right / bottom side (floor division for the
   offset).
Pillow's bilinear resampling uses a support that grows with the reduction factor, so
it averages over the covered source pixels when shrinking (it does not just sample
four neighbours). Padding and the remaining image content do not remove possible
device-specific cues such as burned-in markers or the typical image layout.

Scaling to numbers: uint8 -> float32 / 255, i.e. values in [0, 1]. There are no
statistics to estimate, so there is nothing to fit on the training data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PAD_VALUE = 0  # black, uint8
INTERPOLATION = Image.Resampling.BILINEAR
EXPECTED_MODE = "L"  # 8-bit grayscale, as found for all audited images


class ImageLoadError(Exception):
    """An image could not be used; the message names the file and the reason."""


def normalize_target_size(target_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return (width, height). An int means a square target."""
    width, height = (target_size, target_size) if isinstance(target_size, int) else target_size
    if width < 1 or height < 1:
        raise ValueError(f"target size must be positive, got {target_size!r}")
    return int(width), int(height)


def load_grayscale_image(path: Path, expected_size: tuple[int, int] | None = None) -> Image.Image:
    """Open one PNG read-only and return a detached 8-bit grayscale image.

    expected_size is (width, height) as DECODED in the audited manifest. A mismatch
    is an error; the (partly wrong) Width/Height columns of the original CSV never
    steer any transformation.
    """
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()  # decode fully so that truncated files fail here
            if image.mode != EXPECTED_MODE:
                raise ImageLoadError(f"{path.name}: unexpected image mode {image.mode!r}, "
                                     f"expected {EXPECTED_MODE!r}; no silent conversion is done")
            if expected_size is not None and image.size != tuple(expected_size):
                raise ImageLoadError(f"{path.name}: decoded size {image.size} differs from the "
                                     f"audited size {tuple(expected_size)}")
            return image.copy()
    except ImageLoadError:
        raise
    except FileNotFoundError as exc:
        raise ImageLoadError(f"{path.name}: file not found") from exc
    except (OSError, ValueError, SyntaxError) as exc:  # unreadable, truncated, corrupt
        raise ImageLoadError(f"{path.name}: image cannot be read ({type(exc).__name__}: {exc})") from exc


def resize_with_padding(image: Image.Image, target_size: int | tuple[int, int] = 64) -> Image.Image:
    """Fit the whole image into target_size (width, height) and pad with black.

    See the module docstring for rounding and centring. The target size is a
    parameter so that the same geometry can be reused for other resolutions later.
    """
    width, height = normalize_target_size(target_size)
    fitted = ImageOps.contain(image, (width, height), method=INTERPOLATION)
    if min(fitted.size) < 1:
        raise ValueError(f"image of size {image.size} degenerates to {fitted.size} for "
                         f"target {(width, height)}")
    canvas = Image.new(image.mode, (width, height), color=PAD_VALUE)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def to_unit_float32(image: Image.Image) -> np.ndarray:
    """uint8 grayscale image -> float32 array (height, width) with values in [0, 1]."""
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
    if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
        raise ValueError("pixel values are not finite values in [0, 1]")
    return array


def preprocess_image(path: Path, target_size: int | tuple[int, int] = 64,
                     expected_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load, fit with padding and scale one image: float32 array (height, width)."""
    return to_unit_float32(resize_with_padding(load_grayscale_image(path, expected_size), target_size))


def preprocess_image_set(project_root: Path, relative_paths: list[str],
                         target_size: int | tuple[int, int] = 64,
                         expected_sizes: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Preprocess images in EXACTLY the given order.

    Only the paths handed in are opened; row i of the result belongs to
    relative_paths[i]. The caller decides which images these are (here: the filtered
    train and validation rows), so no image can enter by accident.
    Returns float32 of shape (n, height, width).
    """
    width, height = normalize_target_size(target_size)
    if expected_sizes is not None and len(expected_sizes) != len(relative_paths):
        raise ValueError("expected_sizes and relative_paths must have the same length")
    result = np.empty((len(relative_paths), height, width), dtype=np.float32)
    for index, relative_path in enumerate(relative_paths):
        result[index] = preprocess_image(Path(project_root) / relative_path, (width, height),
                                         None if expected_sizes is None else expected_sizes[index])
    return result


def flatten_for_sklearn(images: np.ndarray) -> np.ndarray:
    """(n, height, width) float32 -> (n, height*width) C-contiguous float64.

    Row-major pixel order (row 0 left to right, then row 1, ...). The float32 -> float64
    conversion is exact; it only gives scikit-learn its preferred contiguous dtype.
    """
    return np.ascontiguousarray(images.reshape(len(images), -1), dtype=np.float64)


def describe_preprocessing(target_size: int | tuple[int, int] = 64) -> dict:
    """Machine-readable description of the fixed preprocessing (stored with models)."""
    import PIL
    width, height = normalize_target_size(target_size)
    return {
        "source": "original PNG at image_relpath (pixels only; no mask, bbox or metadata)",
        "color": "single grayscale channel, 8 bit (mode 'L'); other modes are rejected",
        "target_size_width_height": [width, height],
        "geometry": ("whole image scaled down proportionally to fit (PIL.ImageOps.contain), "
                     "centred on a black canvas; no crop, no stretch"),
        "interpolation": "bilinear (PIL.Image.Resampling.BILINEAR)",
        "side_rounding": "Pillow rounds the scaled side with Python round() (half to even)",
        "centering": "offset = (target - scaled) // 2; an odd remainder lies right/bottom",
        "pad_value_uint8": PAD_VALUE,
        "scaling": "float32(uint8) / 255 -> [0, 1]; fixed, nothing estimated from data",
        "learned_normalization": None,
        "flatten": "row-major (C order), then C-contiguous float64 for scikit-learn (exact conversion)",
        "n_features": width * height,
        "pillow_version": PIL.__version__,
    }


