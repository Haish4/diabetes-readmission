"""Shared data loading and feature engineering.

Used by the notebook, train.py and predict.py so training and serving
apply exactly the same transformations.
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/diabetic_data.csv"

# Discharged to hospice / expired -> these patients cannot be readmitted.
DISCHARGE_EXCLUDE = [11, 13, 14, 19, 20, 21]

NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "total_prior_visits", "n_meds_active", "n_meds_changed",
]

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

CAT_COLS = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MED_COLS

FEATURES = NUM_COLS + CAT_COLS


def icd9_group(code):
    """Map an ICD-9 diagnosis code to a broad disease group."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "unknown"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        v = float(code)
    except ValueError:
        return "other"
    if 250 <= v < 251:
        return "diabetes"
    if 390 <= v <= 459 or v == 785:
        return "circulatory"
    if 460 <= v <= 519 or v == 786:
        return "respiratory"
    if 520 <= v <= 579 or v == 787:
        return "digestive"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def age_to_num(age):
    """'[70-80)' -> 75 (midpoint of the bracket)."""
    lo = int(str(age).strip("[)").split("-")[0])
    return lo + 5


def prepare_features(df):
    """Raw encounter dataframe -> model-ready dataframe (FEATURES columns)."""
    df = df.copy()
    df["age_num"] = df["age"].map(age_to_num)
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    meds = df[MED_COLS]
    df["n_meds_active"] = (meds != "No").sum(axis=1)
    df["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[c] = df[c].astype(str)
    X = df[FEATURES].copy()
    X[CAT_COLS] = X[CAT_COLS].fillna("unknown").astype(str)
    return X


def load_data(path=DATA_PATH):
    """Read the raw CSV ('?' = missing) and drop encounters that cannot be readmitted."""
    df = pd.read_csv(path, na_values="?", keep_default_na=False)
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_EXCLUDE)].reset_index(drop=True)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df
