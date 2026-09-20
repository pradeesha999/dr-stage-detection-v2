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


def setup_gpu(mixed_precision: bool = True) -> str:
    """
    Configure TensorFlow for training and return a one-line device summary.

    * memory growth  - do not grab all 12 GB up front, so notebooks and the
                       Gradio app can share the card;
    * mixed float16  - Ampere tensor cores run fp16 matmuls ~2x faster;
                       Keras keeps the softmax output in float32 for stability.
    Falls back silently to CPU float32 when no GPU is present.
    """
    import tensorflow as tf
    import keras
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus and mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    policy = keras.mixed_precision.global_policy().name
    return f"{len(gpus)} GPU(s): {[g.name for g in gpus]} | policy={policy}"


def save_fig(name: str, dpi: int = 150) -> str:
    """Save the current matplotlib figure into FIG_DIR and return its path."""
    path = config.FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return str(path)
