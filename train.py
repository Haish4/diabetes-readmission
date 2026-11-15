"""Train the final readmission model and save it to model.bin.

Usage:  python train.py
Same logic as the notebook: patient-grouped 60/20/20 split, XGBoost with the
hyper-parameters selected in notebook.ipynb.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CAT_COLS, NUM_COLS, load_data, prepare_features

OUTPUT_FILE = "model.bin"
FLAG_FRACTION = 0.20  # flag the 20% highest-risk discharges

# Best configuration found in notebook.ipynb (validation ROC-AUC 0.688)
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
    subsample=0.7, colsample_bytree=0.5, reg_lambda=1, scale_pos_weight=1,
    tree_method="hist", n_jobs=-1, random_state=1,
)


def make_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", XGBClassifier(**XGB_PARAMS))])


def main():
    print("Loading data...")
    df = load_data()
    X, y, groups = prepare_features(df), df["target"], df["patient_nbr"]

    # identical split to the notebook (grouped by patient to avoid leakage)
    tv_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=1).split(X, y, groups))
    tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=0.25, random_state=1).split(
        X.iloc[tv_idx], y.iloc[tv_idx], groups.iloc[tv_idx]))
    train_idx, val_idx = tv_idx[tr_rel], tv_idx[va_rel]

    print("Training on train split to pick the decision threshold...")
    pipe = make_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
    val_proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    threshold = float(np.quantile(val_proba, 1 - FLAG_FRACTION))
    print(f"validation ROC-AUC = {roc_auc_score(y.iloc[val_idx], val_proba):.4f}, "
          f"threshold (top {FLAG_FRACTION:.0%}) = {threshold:.4f}")

    print("Refitting on train + validation and evaluating on test...")
    pipe = make_pipeline().fit(X.iloc[tv_idx], y.iloc[tv_idx])
    p = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    flagged = p >= threshold
    yt = y.iloc[test_idx].values
    print(f"test ROC-AUC = {roc_auc_score(yt, p):.4f}, PR-AUC = {average_precision_score(yt, p):.4f}")
    print(f"flagged {flagged.mean():.1%} of patients -> precision {yt[flagged].mean():.3f}, "
          f"recall {yt[flagged].sum() / yt.sum():.3f}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"model": pipe, "threshold": threshold}, f)
    print(f"Saved model to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
