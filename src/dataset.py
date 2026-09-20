"""
tf.data input pipeline: loading cached preprocessed PNGs, augmentation and
class-imbalance handling.

Design decisions
----------------
* Images are read from the preprocessed cache (`proc_path`), so the only
  per-epoch CPU work is decoding + augmentation.
* Augmentation uses Keras 3 preprocessing layers applied inside `tf.data`
  (CPU) so the GPU only sees ready batches. Only *retina-safe* transforms
  are used: a fundus photo is rotation/flip invariant, but shear, hue shifts
  or large colour changes would alter the very cues (red haemorrhages,
  yellow exudates, lesion shapes) the model must learn.
* Two imbalance strategies are provided so they can be compared:
    - class weights (loss re-weighting)             -> `class_weights()`
    - oversampling minority classes at batch level  -> `sampling=` argument
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

from . import config

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_augmenter(seed: int = config.SEED) -> keras.Sequential:
    """
    Retina-safe augmentation stack.

    RandomFlip        horizontal + vertical (left/right eyes are mirror images)
    RandomRotation    factor 0.5 -> any angle in [-180, 180] degrees
    RandomZoom        +/-10 %  (simulates different camera field-of-view)
    RandomTranslation +/-5 %   (retina not always perfectly centred)
    RandomBrightness  +/-10 %  (exposure differences between clinics)
    RandomContrast    +/-10 %
    Empty (black) areas created by rotation/zoom are filled with 0, matching
    the black background of the cropped fundus.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomBrightness(0.1, value_range=(0.0, 255.0), seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmenter",
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _decode(path: tf.Tensor, label: tf.Tensor, size: int):
    """Read a PNG, resize to (size, size), return float32 in [0, 255]."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size))          # no-op if already cached at `size`
    return tf.cast(img, tf.float32), label


def _per_class_datasets(df: pd.DataFrame, size: int):
    """One shuffled, repeating dataset per class (used for oversampling)."""
    out = []
    for c in range(config.NUM_CLASSES):
        sub = df[df.label == c]
        ds = tf.data.Dataset.from_tensor_slices(
            (sub.proc_path.values, sub.label.values.astype(np.int32)))
        ds = ds.shuffle(len(sub), seed=config.SEED, reshuffle_each_iteration=True).repeat()
        out.append(ds.map(lambda p, l: _decode(p, l, size), num_parallel_calls=AUTOTUNE))
    return out


def sampling_weights(df: pd.DataFrame, sampling: str) -> np.ndarray:
    """
    Per-class probability of drawing a sample.

    "natural"  -> proportional to class frequency (no oversampling)
    "sqrt"     -> proportional to sqrt(frequency): a compromise that boosts
                  rare classes without flooding batches with duplicates
    "balanced" -> every class equally likely
    """
    freq = df.label.value_counts().reindex(range(config.NUM_CLASSES)).values.astype(float)
    if sampling == "natural":
        w = freq
    elif sampling == "sqrt":
        w = np.sqrt(freq)
    elif sampling == "balanced":
        w = np.ones_like(freq)
    else:
        raise ValueError(sampling)
    return w / w.sum()


def make_dataset(df: pd.DataFrame, batch_size: int = config.BATCH_SIZE,
                 size: int = config.IMG_SIZE, augment: bool = False,
                 shuffle: bool = False, sampling: str = "natural",
                 seed: int = config.SEED) -> tf.data.Dataset:
    """
    Build a batched tf.data pipeline from a split DataFrame.

    augment   apply build_augmenter() to every image (training only)
    shuffle   shuffle order each epoch (training only)
    sampling  "natural" | "sqrt" | "balanced" - see sampling_weights().
              Anything other than "natural" returns an *infinite* dataset;
              pass steps_per_epoch=len(df)//batch_size to model.fit().
    Labels are integer class ids (use sparse categorical cross-entropy).
    """
    if sampling != "natural":
        ds = tf.data.Dataset.sample_from_datasets(
            _per_class_datasets(df, size),
            weights=sampling_weights(df, sampling).tolist(),
            seed=seed, stop_on_empty_dataset=False)
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (df.proc_path.values, df.label.values.astype(np.int32)))
        if shuffle:
            ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: _decode(p, l, size), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def class_weights(df: pd.DataFrame) -> dict:
    """
    sklearn 'balanced' weights: w_c = N / (K * n_c).
    Passed to model.fit(class_weight=...) so mistakes on rare stages cost more.
    """
    w = compute_class_weight("balanced", classes=np.arange(config.NUM_CLASSES), y=df.label.values)
    return {i: float(v) for i, v in enumerate(w)}
