"""
Image preprocessing pipeline for fundus photographs.

Pipeline (each step is a separate function so the notebook can show
before/after evidence for every stage):

    raw BGR/RGB image
      -> crop_black_border   remove the black frame around the retina
      -> resize_square       pad to square, resize to IMG_SIZE (keeps aspect)
      -> apply_clahe         local contrast enhancement (LAB L-channel)
      -> denoise             light Gaussian blur to suppress sensor noise
      -> unsharp_mask        edge enhancement (vessels / lesion boundaries)
      -> uint8 RGB

Mean/std normalisation is *not* done here: it is applied inside the Keras
model via the backbone's own `preprocess_input`, so the saved images stay
viewable and the normalisation always matches the pretrained weights.

Ben Graham's method (Kaggle 2015 winner) is included for comparison.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_rgb(path) -> np.ndarray:
    """Read an image from disk as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------
def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop the black background around the circular retina.

    A pixel is considered background when its grey value is <= `tol`.
    We keep the bounding box of all non-background pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():                       # completely black image
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def resize_square(img: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Pad to a square with black, then resize to (size, size). No distortion."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = np.zeros((side, side, 3), dtype=img.dtype)
    padded[top:top + h, left:left + w] = img
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(padded, (size, size), interpolation=interp)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.

    Applied on the L (lightness) channel of LAB so colours are preserved
    while local contrast of vessels and lesions is boosted. `clip_limit`
    caps amplification to avoid blowing up noise in flat regions.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Light Gaussian smoothing to suppress sensor noise amplified by CLAHE.

    A 3x3 kernel is deliberately small: microaneurysms are only a few pixels
    wide at 224 px and a larger blur (or a median filter) would erase them.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def unsharp_mask(img: np.ndarray, sigma: float = 3.0,
                 amount: float = 1.5) -> np.ndarray:
    """
    Edge enhancement: sharpened = img + amount * (img - blur(img)).

    Emphasises vessel walls, exudate edges and haemorrhage boundaries,
    which are the features that separate DR stages.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def ben_graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Ben Graham's normalisation (Kaggle DR 2015 winner):
    subtract the local average colour so illumination differences vanish.
    out = 4*img - 4*gaussian(img, sigma) + 128
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def circular_mask(img: np.ndarray, radius_frac: float = 0.98) -> np.ndarray:
    """Zero everything outside the inscribed circle (removes blur halo at the rim)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * radius_frac), 1, -1)
    return img * mask[..., None]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess(img: np.ndarray, size: int = config.IMG_SIZE,
               method: str = "clahe", return_steps: bool = False):
    """
    Run the full pipeline on one RGB image.

    method: "clahe"  -> crop, resize, CLAHE, denoise, unsharp   (default)
            "graham" -> crop, resize, Ben Graham, circular mask
            "none"   -> crop, resize only (baseline for ablation)
    return_steps: also return an ordered dict of intermediate images.
    """
    steps = {"original": img}
    x = crop_black_border(img);          steps["cropped"] = x
    x = resize_square(x, size);          steps["resized"] = x

    if method == "clahe":
        x = apply_clahe(x);              steps["clahe"] = x
        x = denoise(x);                  steps["denoised"] = x
        x = unsharp_mask(x);             steps["sharpened"] = x
    elif method == "graham":
        x = ben_graham(x);               steps["graham"] = x
        x = circular_mask(x);            steps["masked"] = x
    elif method != "none":
        raise ValueError(method)

    return (x, steps) if return_steps else x


def preprocess_file(src: str, dst: str, size: int = config.IMG_SIZE,
                    method: str = "clahe") -> str:
    """Read `src`, preprocess, write PNG to `dst`. Returns dst."""
    out = preprocess(read_rgb(src), size, method)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return dst


def preprocess_dataframe(df: pd.DataFrame, size: int = config.IMG_SIZE,
                         method: str = "clahe", workers: int = 8,
                         limit: int | None = None) -> pd.DataFrame:
    """
    Preprocess every image in `df` once and cache it at `df.proc_path`
    (set by data.load_splits -> config.PROC_DIR/<split>/<class>/<id>.png).

    Doing this offline means training epochs only pay for augmentation,
    not CLAHE/sharpening, which makes training several times faster.
    Already-cached files are skipped, so re-running is cheap.
    """
    df = df.copy() if limit is None else df.head(limit).copy()
    todo = df[[not Path(p).exists() for p in df.proc_path]]
    print(f"{len(df) - len(todo):,} already cached, {len(todo):,} to process")

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(lambda r: preprocess_file(r.image_path, r.proc_path, size, method),
                         todo.itertuples()), total=len(todo)))
    return df


# ---------------------------------------------------------------------------
# Quality metrics (used in the notebook to prove preprocessing helps)
# ---------------------------------------------------------------------------
def image_quality(img: np.ndarray) -> dict:
    """
    Simple, objective image-quality numbers:
      contrast  = std of grey levels (higher = more contrast)
      sharpness = variance of Laplacian (higher = more edge detail)
      entropy   = information content of grey histogram
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return dict(contrast=float(gray.std()),
                sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                entropy=float(-(p * np.log2(p)).sum()))
