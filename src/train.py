"""
Training loop, callbacks and experiment bookkeeping.

`run_experiment()` is the single entry point used by notebook 04. Every run:
  * builds datasets according to the balancing strategy,
  * trains in one or two phases (head only -> fine-tune),
  * evaluates on the validation set with macro-F1 and quadratic-weighted kappa,
  * writes   outputs/metrics/<run>.json      (final numbers + config)
             outputs/metrics/<run>_history.csv (per-epoch curves)
             outputs/models/<run>.keras        (best weights)
so results survive the Kaggle session and can be compared in a table.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import callbacks as kc
from sklearn.metrics import cohen_kappa_score, f1_score

from . import config, dataset, model as M

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = config.WORK_DIR / "metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Validation metrics that Keras cannot compute natively on sparse labels
# ---------------------------------------------------------------------------
class ValMetrics(kc.Callback):
    """
    After each epoch predict the whole validation set and log
    val_macro_f1 and val_qwk (quadratic weighted kappa - the standard DR
    metric, it rewards being *close* on the ordinal stage scale).
    Other callbacks (checkpoint / early stopping) can then monitor them.
    """

    def __init__(self, val_ds, y_true):
        super().__init__()
        self.val_ds, self.y_true = val_ds, np.asarray(y_true)

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0)
        pred = probs.argmax(1)
        logs["val_macro_f1"] = float(f1_score(self.y_true, pred, average="macro"))
        logs["val_qwk"] = float(cohen_kappa_score(self.y_true, pred, weights="quadratic"))
        print(f"   val_macro_f1={logs['val_macro_f1']:.4f}  val_qwk={logs['val_qwk']:.4f}")


def make_callbacks(run_name: str, val_ds, y_val, monitor: str = "val_qwk",
                   patience: int = 5) -> list:
    """Standard callback stack: metrics, checkpoint, early stop, LR schedule, CSV log."""
    ckpt_path = config.MODEL_DIR / f"{run_name}.keras"
    return [
        ValMetrics(val_ds, y_val),                       # must run first
        kc.ModelCheckpoint(str(ckpt_path), monitor=monitor, mode="max",
                           save_best_only=True, verbose=0),
        kc.EarlyStopping(monitor=monitor, mode="max", patience=patience,
                         restore_best_weights=True, verbose=1),
        kc.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                             min_lr=1e-7, verbose=1),
        kc.CSVLogger(str(METRIC_DIR / f"{run_name}_history.csv"), append=True),
    ]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(run_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   backbone: str = "efficientnetb0",
                   balancing: str = "class_weights",      # none | class_weights | sqrt | balanced
                   img_size: int = config.IMG_SIZE,
                   batch_size: int = config.BATCH_SIZE,
                   epochs_head: int = 5, epochs_finetune: int = 10,
                   lr_head: float = 1e-3, lr_finetune: float = 1e-4,
                   unfreeze_fraction: float = 0.3, dropout: float = 0.3,
                   label_smoothing: float = 0.0, patience: int = 5,
                   baseline_cnn: bool = False, notes: str = "") -> dict:
    """
    Train one configuration end-to-end and persist everything.
    Returns a dict with the config and best validation metrics.
    """
    t0 = time.time()
    keras.utils.set_random_seed(config.SEED)

    # ---- data ------------------------------------------------------------
    sampling = balancing if balancing in ("sqrt", "balanced") else "natural"
    train_ds = dataset.make_dataset(train_df, batch_size, img_size, augment=True,
                                    shuffle=True, sampling=sampling)
    val_ds = dataset.make_dataset(val_df, batch_size, img_size)
    steps = len(train_df) // batch_size if sampling != "natural" else None
    class_weight = dataset.class_weights(train_df) if balancing == "class_weights" else None

    # ---- model -----------------------------------------------------------
    if baseline_cnn:
        model = M.build_baseline_cnn(img_size)
    else:
        model = M.build_transfer_model(backbone, img_size, dropout=dropout)
    loss = keras.losses.SparseCategoricalCrossentropy() if label_smoothing == 0 else \
        _smoothed_sparse_ce(label_smoothing)

    cbs = make_callbacks(run_name, val_ds, val_df.label.values, patience=patience)
    history = {}

    def _fit(lr, epochs, tag):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
        print(f"\n[{run_name}] {tag}: lr={lr} epochs={epochs} "
              f"params={M.count_params(model)}")
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                      steps_per_epoch=steps, class_weight=class_weight,
                      callbacks=cbs, verbose=2)
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)

    # ---- phase 1: head only ------------------------------------------------
    if epochs_head > 0:
        _fit(lr_head, epochs_head, "phase 1 (frozen backbone)")

    # ---- phase 2: fine-tune -------------------------------------------------
    if epochs_finetune > 0 and not baseline_cnn:
        n = M.unfreeze_top(model, unfreeze_fraction)
        print(f"unfroze {n} backbone layers (top {unfreeze_fraction:.0%})")
        _fit(lr_finetune, epochs_finetune, "phase 2 (fine-tune)")

    # ---- final validation numbers (best weights restored by EarlyStopping) ---
    probs = model.predict(val_ds, verbose=0)
    pred = probs.argmax(1)
    y = val_df.label.values
    result = {
        "run": run_name, "backbone": "baseline_cnn" if baseline_cnn else backbone,
        "balancing": balancing, "img_size": img_size, "batch_size": batch_size,
        "epochs_head": epochs_head, "epochs_finetune": epochs_finetune,
        "lr_head": lr_head, "lr_finetune": lr_finetune,
        "unfreeze_fraction": unfreeze_fraction, "dropout": dropout,
        "label_smoothing": label_smoothing, "notes": notes,
        "epochs_run": len(history.get("loss", [])),
        "params": M.count_params(model),
        "val_accuracy": float((pred == y).mean()),
        "val_macro_f1": float(f1_score(y, pred, average="macro")),
        "val_qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (METRIC_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(history).to_csv(METRIC_DIR / f"{run_name}_curves.csv", index_label="epoch")
    print(f"\n[{run_name}] done in {result['minutes']} min | "
          f"acc={result['val_accuracy']:.4f} macroF1={result['val_macro_f1']:.4f} "
          f"QWK={result['val_qwk']:.4f}")
    return result


def _smoothed_sparse_ce(eps: float):
    """Sparse CE with label smoothing (Keras only offers it for one-hot)."""
    cce = keras.losses.CategoricalCrossentropy(label_smoothing=eps)

    def loss(y_true, y_pred):
        y_1h = keras.ops.one_hot(keras.ops.cast(keras.ops.squeeze(y_true, -1)
                                                if len(y_true.shape) > 1 else y_true, "int32"),
                                 config.NUM_CLASSES)
        return cce(y_1h, y_pred)
    return loss


def results_table() -> pd.DataFrame:
    """Collect every outputs/metrics/*.json into one comparison table."""
    rows = [json.loads(p.read_text()) for p in sorted(METRIC_DIR.glob("*.json"))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["params_M"] = df.params.apply(lambda p: round(p["total"] / 1e6, 1))
    cols = ["run", "backbone", "balancing", "img_size", "epochs_run", "params_M",
            "val_accuracy", "val_macro_f1", "val_qwk", "minutes", "notes"]
    return df[cols].sort_values("val_qwk", ascending=False).reset_index(drop=True)


def load_curves(run_name: str) -> pd.DataFrame:
    return pd.read_csv(METRIC_DIR / f"{run_name}_curves.csv")
