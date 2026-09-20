"""
Dataset indexing and patient-level splitting.

The Kaggle 2015 dataset stores one image per eye, named <patient>_<left|right>.
Both eyes of one patient must land in the same split, otherwise the model can
memorise a patient in training and be tested on their other eye (data leakage).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def build_dataframe() -> pd.DataFrame:
    """
    Walk the class folders and return a DataFrame with one row per image:
    image_path, image_id, patient_id, eye, class_name, label.
    """
    rows = []
    for class_name, label in config.CLASS_TO_LABEL.items():
        folder = config.IMAGE_DIR / class_name
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image_id = path.stem                         # e.g. 10_left
            patient_id, eye = image_id.rsplit("_", 1)    # "10", "left"
            rows.append(
                dict(image_path=str(path), image_id=image_id,
                     patient_id=patient_id, eye=eye,
                     class_name=class_name, label=label)
            )
    df = pd.DataFrame(rows)
    # Cross-check folder labels against the official label file
    labels = pd.read_csv(config.LABELS_CSV)
    merged = df.merge(labels, left_on="image_id", right_on="image", how="left")
    mismatch = (merged["level"] != merged["label"]).sum()
    if mismatch:
        print(f"WARNING: {mismatch} folder labels disagree with trainLabels.csv")
    return df


def patient_split(df: pd.DataFrame,
                  train_frac: float = config.TRAIN_FRAC,
                  val_frac: float = config.VAL_FRAC,
                  seed: int = config.SEED):
    """
    Stratified split at *patient* level.

    Each patient is assigned the max severity of their two eyes for
    stratification, then patients (not images) are split, so both eyes always
    stay together. Returns (train_df, val_df, test_df).
    """
    patients = (df.groupby("patient_id")["label"].max()
                  .reset_index().rename(columns={"label": "strat"}))

    train_p, rest_p = train_test_split(
        patients, train_size=train_frac, stratify=patients["strat"],
        random_state=seed)
    rel_val = val_frac / (1 - train_frac)
    val_p, test_p = train_test_split(
        rest_p, train_size=rel_val, stratify=rest_p["strat"],
        random_state=seed)

    def pick(p):
        return df[df["patient_id"].isin(p["patient_id"])].reset_index(drop=True)

    train_df, val_df, test_df = pick(train_p), pick(val_p), pick(test_p)
    # Guarantee no patient appears in two splits
    assert not set(train_df.patient_id) & set(val_df.patient_id)
    assert not set(train_df.patient_id) & set(test_df.patient_id)
    assert not set(val_df.patient_id) & set(test_df.patient_id)
    return train_df, val_df, test_df


SPLIT_COLS = ["image_id", "patient_id", "eye", "class_name", "label"]


def add_paths(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Attach machine-specific paths to a split DataFrame.

    image_path -> raw image under config.IMAGE_DIR
    proc_path  -> cached preprocessed image under config.PROC_DIR/<split>/
    Paths are rebuilt on every machine so the CSVs stay portable.
    """
    df = df.copy()
    df["image_path"] = [str(config.IMAGE_DIR / c / f"{i}.png")
                        for c, i in zip(df.class_name, df.image_id)]
    df["proc_path"] = [str(config.PROC_DIR / split / c / f"{i}.png")
                       for c, i in zip(df.class_name, df.image_id)]
    return df


def save_splits(train_df, val_df, test_df) -> None:
    """Persist the split (IDs + labels only, no absolute paths) so every
    notebook and every machine uses identical sets."""
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        d[SPLIT_COLS].to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
    print(f"splits saved to {config.SPLIT_DIR}")


def load_splits():
    """Load train/val/test CSVs and attach paths valid on this machine."""
    return tuple(add_paths(pd.read_csv(config.SPLIT_DIR / f"{n}.csv",
                                       dtype={"patient_id": str}), n)
                 for n in ("train", "val", "test"))
