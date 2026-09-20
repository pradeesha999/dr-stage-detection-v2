"""
Central configuration for the DR project.

Every notebook imports this so paths, image size, class names and the
random seed are defined in exactly one place. The data root is detected
automatically so the same code runs on Kaggle and on a local clone.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
KAGGLE_INPUT = Path("/kaggle/input/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized")
LOCAL_INPUT = Path(__file__).resolve().parents[2] / "archive"


def find_data_root() -> Path:
    """Return the folder that contains trainLabels.csv and colored_images/."""
    for candidate in (KAGGLE_INPUT, LOCAL_INPUT):
        if (candidate / "trainLabels.csv").exists():
            return candidate
    # Fall back to searching /kaggle/input in case the dataset slug differs
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for csv in kaggle_root.rglob("trainLabels.csv"):
            return csv.parent
    raise FileNotFoundError("Could not locate dataset (trainLabels.csv not found).")


DATA_ROOT = find_data_root()
IMAGE_DIR = DATA_ROOT / "colored_images" / "colored_images"
LABELS_CSV = DATA_ROOT / "trainLabels.csv"

# Where we write things: <repo>/outputs. On Kaggle the repo is cloned into
# /kaggle/working/dr_project, so this is writable there too.
WORK_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIG_DIR = WORK_DIR / "figures"
MODEL_DIR = WORK_DIR / "models"
SPLIT_DIR = WORK_DIR / "splits"
PROC_DIR = WORK_DIR / "processed"      # cached preprocessed images
for _d in (FIG_DIR, MODEL_DIR, SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes (International Clinical DR severity scale, as used by Kaggle 2015)
# ---------------------------------------------------------------------------
# Folder name  -> numeric label
CLASS_TO_LABEL = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}
CLASS_NAMES = [LABEL_TO_CLASS[i] for i in range(len(LABEL_TO_CLASS))]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image / training hyper-parameters (defaults; notebooks may override)
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Split fractions (by patient, see data.py)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
