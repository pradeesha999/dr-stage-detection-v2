"""
Evaluation helpers: loading the final model, batch prediction, metrics,
and Grad-CAM. Used by notebook 05 and the Gradio demo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)

from . import config, dataset, model as M

METRIC_DIR = config.WORK_DIR / "metrics"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def final_run_name() -> str:
    p = METRIC_DIR / "final_run.txt"
    return p.read_text().strip() if p.exists() else "E4_final"


def load_run_config(run: str) -> dict:
    return json.loads((METRIC_DIR / f"{run}.json").read_text())


def load_model_fp32(run: str | None = None) -> tuple[keras.Model, dict]:
    """
    Load `outputs/models/<run>.keras` and return a float32 copy.

    Training used mixed precision (float16 activations). Rebuilding the same
    architecture under a float32 policy and copying the weights gives a model
    that runs anywhere (CPU demo app, Grad-CAM gradients) with no fp16 quirks.
    """
    run = run or final_run_name()
    cfg = load_run_config(run)
    path = config.MODEL_DIR / f"{run}.keras"
    if not path.exists():
        raise FileNotFoundError(f"{path} - copy the trained model into outputs/models/")
    keras.mixed_precision.set_global_policy("float32")
    trained = keras.models.load_model(path, compile=False)
    if cfg["backbone"] == "baseline_cnn":
        fresh = M.build_baseline_cnn(cfg["img_size"])
    else:
        fresh = M.build_transfer_model(cfg["backbone"], cfg["img_size"],
                                       dropout=cfg["dropout"], weights=None)
    fresh.set_weights(trained.get_weights())
    return fresh, cfg


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_df(model: keras.Model, df: pd.DataFrame, batch_size: int = 64,
               img_size: int = config.IMG_SIZE, tta: bool = False) -> np.ndarray:
    """
    Class probabilities for every row of `df` (reads `proc_path`).
    tta=True averages predictions over 4 flips (test-time augmentation).
    """
    ds = dataset.make_dataset(df, batch_size, img_size)
    if not tta:
        return model.predict(ds, verbose=0)
    probs = np.zeros((len(df), config.NUM_CLASSES), dtype=np.float32)
    for flip in (lambda x: x, tf.image.flip_left_right, tf.image.flip_up_down,
                 lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))):
        probs += model.predict(ds.map(lambda x, y: (flip(x), y)), verbose=0)
    return probs / 4


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summary_metrics(y_true, probs) -> dict:
    """Headline numbers used in the report tables."""
    y_pred = probs.argmax(1)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_precision": float(classification_report(y_true, y_pred, output_dict=True, zero_division=0)["macro avg"]["precision"]),
        "macro_recall": float(classification_report(y_true, y_pred, output_dict=True, zero_division=0)["macro avg"]["recall"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "auc_ovr_macro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
        "within_one_stage": float((np.abs(y_pred - y_true) <= 1).mean()),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    df["support"] = df["support"].astype(int)
    return df.round(3)


def referable_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    """
    Binary screening view used in clinical DR literature:
    referable DR = Moderate or worse (stage >= 2).
    """
    p_ref = probs[:, 2:].sum(1)
    y_ref = (np.asarray(y_true) >= 2).astype(int)
    pred = (p_ref >= threshold).astype(int)
    tp = int(((pred == 1) & (y_ref == 1)).sum()); tn = int(((pred == 0) & (y_ref == 0)).sum())
    fp = int(((pred == 1) & (y_ref == 0)).sum()); fn = int(((pred == 0) & (y_ref == 1)).sum())
    return {"threshold": threshold, "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "auc": float(roc_auc_score(y_ref, p_ref)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def confusion(y_true, y_pred, normalize: bool = False) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    if normalize:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    return cm


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping on the backbone's last
    feature map. Shows *where* the network looked when predicting a stage;
    for DR we expect heat on haemorrhages / exudates, not on the optic disc
    or the image border.
    """

    def __init__(self, model: keras.Model):
        # The head is a linear chain (norm -> backbone -> GAP -> ... -> softmax),
        # so re-thread the layers to expose the backbone feature map as an output.
        layers_ = model.layers
        idx = next(i for i, l in enumerate(layers_) if isinstance(l, keras.Model))
        inp = keras.Input(model.input_shape[1:], name="cam_in")
        x = inp
        for l in layers_[1:idx]:                 # skip InputLayer
            x = l(x)
        feat = layers_[idx](x, training=False)
        y = feat
        for l in layers_[idx + 1:]:
            y = l(y)
        self.grad_model = keras.Model(inp, [feat, y])

    def __call__(self, img: np.ndarray, class_index: int | None = None):
        """img: (H,W,3) float [0,255]. Returns (heatmap in [0,1] at input size, class, prob)."""
        x = tf.convert_to_tensor(img[None].astype("float32"))
        with tf.GradientTape() as tape:
            feat, probs = self.grad_model(x, training=False)
            if class_index is None:
                class_index = int(tf.argmax(probs[0]))
            score = probs[:, class_index]
        grads = tape.gradient(score, feat)                       # (1,h,w,c)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., None], img.shape[:2])[..., 0].numpy()
        return cam, class_index, float(probs[0, class_index])


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a jet-coloured heatmap onto an RGB uint8 image."""
    import cv2
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * img.astype("float32")).clip(0, 255).astype("uint8")
