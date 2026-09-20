"""
tf.data input pipeline, retina-safe augmentation and class balancing.

Images are read from the preprocessed cache (outputs/processed, built by
notebook 02) so an epoch only pays for PNG decoding + augmentation.

Augmentation policy (all label-preserving for fundus images):
    flip h/v          the retina has no canonical orientation
    rotation, any     same reason - lesions look identical at any angle
    zoom  +-10 %      camera framing varies between photos
    brightness +-10 % exposure differences seen in EDA
    contrast   +-10 % idem
Deliberately *not* used: shear (distorts the circular disc), hue/colour
shifts (haemorrhage vs exudate is partly a colour decision), heavy blur
(erases microaneurysms).

Class imbalance (No_DR is 73 % of the data) is handled by a combination of
partial oversampling and class-weighted loss, see `balance_table`.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from . import config

AUTOTUNE = tf.data.AUTOTUNE
BALANCE_MODES = ("none", "weights", "oversample")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_augmenter(rotation: float = 1.0, zoom: float = 0.10,
                    brightness: float = 0.10, contrast: float = 0.10,
                    seed: int = config.SEED) -> keras.Sequential:
    """
    Return a Keras model that applies random, per-image augmentation.

    Called on a *batch* inside the tf.data pipeline with training=True, so
    every image in the batch gets its own random parameters. Empty corners
    created by rotation / zoom are filled with black, matching the natural
    background of a fundus photo.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(rotation, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomZoom(zoom, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomBrightness(brightness, value_range=(0, 255), seed=seed),
        layers.RandomContrast(contrast, seed=seed),
    ], name="augment")


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------
def balance_table(df: pd.DataFrame, mode: str = "oversample") -> pd.DataFrame:
    """
    Describe, per class, what a balancing mode does to one training epoch.

    mode = "none"        natural sampling, all weights 1
           "weights"     natural sampling, loss weight  N / (K * n_c)
           "oversample"  sample classes with probability p_c ~ sqrt(n_c)
                         (partial oversampling) and weight the loss by
                         (1/K) / p_c so every class contributes equally.

    Columns: n_images, sampling_prob, images_per_epoch, class_weight,
    loss_share (fraction of the total loss each class accounts for).
    Splitting the correction between sampling and weighting keeps both the
    repetition factor (~6x for Proliferate_DR instead of ~36x) and the
    weight magnitude moderate, which trains more stably than either alone.
    """
    if mode not in BALANCE_MODES:
        raise ValueError(f"mode must be one of {BALANCE_MODES}")
    counts = df["label"].value_counts().sort_index().reindex(range(config.NUM_CLASSES), fill_value=0)
    n = counts.values.astype(float)
    N, K = n.sum(), len(n)

    if mode == "oversample":
        p = np.sqrt(n) / np.sqrt(n).sum()
    else:
        p = n / N
    if mode == "none":
        w = np.ones(K)
    else:
        w = (1.0 / K) / p
    w = w / (p * w).sum()             # normalise so the mean sample weight is 1

    return pd.DataFrame({
        "class": config.CLASS_NAMES,
        "n_images": n.astype(int),
        "sampling_prob": p,
        "images_per_epoch": (p * N).round().astype(int),
        "class_weight": w,
        "loss_share": p * w,
    }).set_index("class")


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def _read(path, label):
    """Read the raw PNG bytes (cheap to keep in RAM: ~80 kB per image)."""
    return tf.io.read_file(path), label


def _decode(png, label):
    """Decode PNG bytes -> float32 [0,255] tensor, one-hot label, int label."""
    img = tf.io.decode_png(png, channels=3)
    img = tf.ensure_shape(img, (config.IMG_SIZE, config.IMG_SIZE, 3))
    img = tf.cast(img, tf.float32)
    return img, tf.one_hot(label, config.NUM_CLASSES), label


_SOURCE_CACHE: dict = {}


def _source(paths, labels, cache: bool) -> tf.data.Dataset:
    """
    (path, label) -> (png_bytes, label), optionally cached in RAM.

    Cached sources are memoised per file list so every experiment in a
    notebook shares one copy of the bytes. Without this each run allocated
    its own ~2.3 GB cache, which TensorFlow never freed -> swapping after
    a handful of runs.
    """
    key = (len(paths), paths[0], paths[-1], hash(tuple(paths)))
    if cache and key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(_read, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = _SOURCE_CACHE[key] = ds.cache()
    return ds


def make_dataset(df: pd.DataFrame, batch_size: int = config.BATCH_SIZE,
                 training: bool = False, augment: bool = False,
                 balance: str = "none", cache: bool = True,
                 seed: int = config.SEED) -> tf.data.Dataset:
    """
    Build a batched, prefetched dataset from a split DataFrame.

    training=False -> yields (image, one_hot) in file order, no shuffling
    training=True  -> shuffled, yields (image, one_hot, sample_weight) so the
                      loss is class-weighted; with balance="oversample" the
                      dataset is infinite - use steps_per_epoch(df, batch).
    cache=True     -> the *encoded* PNG bytes are kept in RAM after the first
                      pass (train ~1.9 GB, val ~0.4 GB). The project lives on
                      a spinning disk where random reads of 24k small files
                      cap the pipeline at ~170 img/s; cached it is CPU-bound.
    Images come out as float32 in [0, 255]; the model normalises internally.
    """
    paths = df["proc_path"].values
    labels = df["label"].values.astype(np.int32)

    if training and balance == "oversample":
        tab = balance_table(df, "oversample")
        per_class = []
        for c in range(config.NUM_CLASSES):
            m = labels == c
            d = _source(paths[m], labels[m], cache)
            per_class.append(d.shuffle(int(m.sum()), seed=seed + c,
                                       reshuffle_each_iteration=True).repeat())
        ds = tf.data.Dataset.sample_from_datasets(
            per_class, weights=tab["sampling_prob"].values.astype(np.float32), seed=seed)
    else:
        ds = _source(paths, labels, cache)
        if training:
            ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)

    if training:
        w_table = tf.constant(balance_table(df, balance)["class_weight"].values, tf.float32)
        ds = ds.map(lambda x, y, l: (x, y, tf.gather(w_table, l)), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y, l: (x, y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training)

    if augment:
        aug = build_augmenter(seed=seed)

        @tf.autograph.experimental.do_not_convert
        def _aug(x, y, w):
            return aug(x, training=True), y, w
        ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def steps_per_epoch(df: pd.DataFrame, batch_size: int = config.BATCH_SIZE) -> int:
    """One epoch = as many batches as the split has images (for infinite datasets)."""
    return len(df) // batch_size
