"""Small helpers shared by all notebooks."""
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Fix every RNG we use so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_fig(name: str, dpi: int = 150) -> str:
    """Save the current matplotlib figure into FIG_DIR and return its path."""
    path = config.FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return str(path)
