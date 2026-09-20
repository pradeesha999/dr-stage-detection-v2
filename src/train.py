"""
Experiment runner: trains a model, logs everything, appends one row to
outputs/runs/experiments.csv so the report can tabulate every run.

Per run we write
    outputs/runs/<name>/config.json    all hyper-parameters
    outputs/runs/<name>/history.json   per-epoch metrics (+ stage boundary)
    outputs/runs/<name>/log.csv        Keras CSVLogger
    outputs/models/<name>.keras        best checkpoint (by val macro-F1)

Two-stage transfer learning (`run_transfer`):
    A. backbone frozen, train the new head at a high LR   (few epochs)
    B. unfreeze top blocks, low LR, ReduceLROnPlateau + EarlyStopping
"""
import json
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from . import augment, config, model as M

EXPERIMENTS_CSV = config.RUN_DIR / "experiments.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_callbacks(name: str, early_stop: bool = True, patience: int = 5) -> list:
    """
    ModelCheckpoint  keep the epoch with the best *validation macro-F1*
                     (accuracy is misleading with 73 % No_DR).
    ReduceLROnPlateau  x0.3 after 2 flat epochs of val_loss.
    EarlyStopping    stop after `patience` flat epochs of val_loss.
    CSVLogger        raw per-epoch log.
    """
    run_dir = config.RUN_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_DIR / f"{name}.keras", monitor="val_macro_f1",
            mode="max", save_best_only=True, verbose=0),
        keras.callbacks.CSVLogger(run_dir / "log.csv", append=True),
    ]
    if early_stop:
        cbs += [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=2, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          verbose=1),
        ]
    return cbs


def _merge_histories(hists: list) -> dict:
    """Concatenate Keras History objects from consecutive fit() calls."""
    out = {}
    for h in hists:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def _summarise(name: str, params: dict, history: dict, elapsed: float,
               stage_boundary: int) -> dict:
    """Best-epoch numbers for the experiments table."""
    f1 = np.array(history["val_macro_f1"])
    best = int(f1.argmax())
    n = len(f1)
    return {
        "run": name, **params,
        "epochs_run": n, "stage_boundary": stage_boundary,
        "best_epoch": best + 1,
        "best_val_f1": round(float(f1[best]), 4),
        "val_acc_at_best": round(history["val_acc"][best], 4),
        "val_loss_at_best": round(history["val_loss"][best], 4),
        "train_acc_at_best": round(history["acc"][best], 4),
        "train_min": round(elapsed / 60, 1),
        "sec_per_epoch": round(elapsed / max(n, 1), 1),
    }


def _save_run(name: str, params: dict, history: dict, stage_boundary: int,
              summary: dict) -> None:
    run_dir = config.RUN_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(params, indent=2, default=str))
    (run_dir / "history.json").write_text(
        json.dumps({"stage_boundary": stage_boundary, **history}, indent=1))
    # experiments.csv: replace an existing row with the same run name
    row = pd.DataFrame([summary])
    if EXPERIMENTS_CSV.exists():
        old = pd.read_csv(EXPERIMENTS_CSV)
        old = old[old["run"] != name]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(EXPERIMENTS_CSV, index=False)
    print(f"[{name}] best val macro-F1 {summary['best_val_f1']:.4f} "
          f"(epoch {summary['best_epoch']}), acc {summary['val_acc_at_best']:.4f}, "
          f"{summary['train_min']} min")


def load_history(name: str) -> dict:
    return json.loads((config.RUN_DIR / name / "history.json").read_text())


def load_experiments() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()


def _already_done(name: str, force: bool):
    """Return the saved summary row if this run finished earlier (resumable notebooks)."""
    exp = load_experiments()
    if force or exp.empty or name not in exp["run"].values:
        return None
    if not (config.RUN_DIR / name / "history.json").exists():
        return None
    row = exp[exp["run"] == name].iloc[0].to_dict()
    print(f"[{name}] already done (val macro-F1 {row['best_val_f1']:.4f}) - skipping; force=True to rerun")
    return row


def _datasets(train_df, val_df, batch_size, augment_on, balance):
    train_ds = augment.make_dataset(train_df, batch_size, training=True,
                                    augment=augment_on, balance=balance)
    val_ds = augment.make_dataset(val_df, batch_size)
    steps = augment.steps_per_epoch(train_df, batch_size) if balance == "oversample" else None
    return train_ds, val_ds, steps


# ---------------------------------------------------------------------------
# experiment entry points
# ---------------------------------------------------------------------------
def run_baseline(name: str, train_df, val_df, *, epochs: int = 10,
                 lr: float = 1e-3, dropout: float = 0.3, augment_on: bool = True,
                 balance: str = "oversample", batch_size: int = config.BATCH_SIZE,
                 verbose: int = 1, force: bool = False) -> dict:
    """Train the from-scratch CNN in a single stage."""
    done = _already_done(name, force)
    if done:
        return done
    keras.utils.clear_session()
    params = dict(model="baseline_cnn", backbone="-", balance=balance,
                  augment=augment_on, dropout=dropout, lr_head=lr, lr_ft=None,
                  unfreeze=None, batch_size=batch_size)
    train_ds, val_ds, steps = _datasets(train_df, val_df, batch_size, augment_on, balance)

    net = M.compile_model(M.build_baseline_cnn(dropout=dropout), lr)
    params.update(M.count_params(net))

    t0 = time.time()
    h = net.fit(train_ds, validation_data=val_ds, epochs=epochs,
                steps_per_epoch=steps, callbacks=make_callbacks(name), verbose=verbose)
    history = _merge_histories([h])
    summary = _summarise(name, params, history, time.time() - t0, stage_boundary=0)
    _save_run(name, params, history, 0, summary)
    return summary


def run_transfer(name: str, train_df, val_df, *, backbone: str = "EfficientNetB0",
                 epochs_head: int = 5, epochs_ft: int = 20,
                 lr_head: float = 1e-3, lr_ft: float = 1e-4, unfreeze: float = 0.3,
                 dropout: float = 0.3, augment_on: bool = True,
                 balance: str = "oversample", batch_size: int = config.BATCH_SIZE,
                 patience: int = 5, verbose: int = 1, force: bool = False) -> dict:
    """
    Two-stage transfer learning. `epochs_ft=0` runs the head-only stage
    (used for the cheap ablations / hyper-parameter sweeps).
    Finished runs are skipped unless force=True, so the notebook can resume.
    """
    done = _already_done(name, force)
    if done:
        return done
    keras.utils.clear_session()
    params = dict(model="transfer", backbone=backbone, balance=balance,
                  augment=augment_on, dropout=dropout, lr_head=lr_head,
                  lr_ft=lr_ft if epochs_ft else None,
                  unfreeze=unfreeze if epochs_ft else None, batch_size=batch_size)
    train_ds, val_ds, steps = _datasets(train_df, val_df, batch_size, augment_on, balance)

    net = M.build_transfer_model(backbone, dropout=dropout)
    hists, t0 = [], time.time()

    # --- stage A: frozen backbone, train the head --------------------------
    M.compile_model(net, lr_head)
    print(f"[{name}] stage A: head only, {M.count_params(net)}")
    hists.append(net.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
                         steps_per_epoch=steps, verbose=verbose,
                         callbacks=make_callbacks(name, early_stop=False)))

    # --- stage B: unfreeze top blocks, low LR --------------------------------
    if epochs_ft:
        n_unfrozen = M.unfreeze_top(net, unfreeze)
        M.compile_model(net, lr_ft)
        print(f"[{name}] stage B: {n_unfrozen} layers unfrozen, {M.count_params(net)}")
        hists.append(net.fit(train_ds, validation_data=val_ds,
                             epochs=epochs_head + epochs_ft, initial_epoch=epochs_head,
                             steps_per_epoch=steps, verbose=verbose,
                             callbacks=make_callbacks(name, patience=patience)))
    params.update(M.count_params(net))

    history = _merge_histories(hists)
    summary = _summarise(name, params, history, time.time() - t0, stage_boundary=epochs_head)
    _save_run(name, params, history, epochs_head, summary)
    return summary
