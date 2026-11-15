"""Shared data loading and feature engineering.

Used by the notebook, train.py and predict.py so training and serving
apply exactly the same transformations.
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/diabetic_data.csv"

# Discharged to hospice / expired -> these patients cannot be readmitted.
DISCHARGE_EXCLUDE = [11, 13, 14, 19, 20, 21]

# Patient-history features: computed from the patient's EARLIER encounters only
# (encounter_id increases with time). At serving time the client sends them.
HIST_COLS = [
    "n_prior_enc",      # number of earlier hospital encounters
    "prior_lt30_cnt",   # earlier encounters that were readmitted within 30 days
    "prior_any_cnt",    # earlier encounters that were readmitted at all
    "prev_lt30",        # 1 if the immediately preceding encounter was readmitted <30 days
    "prev_any",         # 1 if the immediately preceding encounter was readmitted at all
    "prior_days",       # total days spent in hospital in earlier encounters
]

BASE_NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "total_prior_visits", "n_meds_active", "n_meds_changed",
    "inpatient_x_emerg", "meds_per_day", "labs_per_day", "has_prior_inpatient",
]
NUM_COLS = BASE_NUM_COLS + HIST_COLS

# All 21 drug columns (used to count active / changed drugs)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
# Only drugs prescribed in >= 1% of encounters are used as categorical features
COMMON_MED_COLS = [
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin",
]

CAT_COLS = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "diag_1_3d", "diag_2_3d", "diag_3_3d",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + COMMON_MED_COLS

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


def icd9_3digit(code):
    """'250.01' -> '250' (finer than the disease group)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "unknown"
    return str(code).split(".")[0]


def age_to_num(age):
    """'[70-80)' -> 75 (midpoint of the bracket)."""
    lo = int(str(age).strip("[)").split("-")[0])
    return lo + 5


def add_patient_history(raw):
    """Add HIST_COLS using only each patient's earlier encounters (no future leakage).

    Needs the full raw table (all encounters of every patient) and the
    `readmitted` label of earlier encounters. Rows are sorted by patient and time.
    """
    raw = raw.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    lt30 = (raw["readmitted"] == "<30").astype(int)
    anyr = (raw["readmitted"] != "NO").astype(int)
    g = raw.assign(_lt30=lt30, _any=anyr).groupby("patient_nbr")
    raw["n_prior_enc"] = g.cumcount()
    raw["prior_lt30_cnt"] = g["_lt30"].cumsum() - lt30
    raw["prior_any_cnt"] = g["_any"].cumsum() - anyr
    raw["prev_lt30"] = g["_lt30"].shift(1).fillna(0).astype(int)
    raw["prev_any"] = g["_any"].shift(1).fillna(0).astype(int)
    raw["prior_days"] = g["time_in_hospital"].cumsum() - raw["time_in_hospital"]
    return raw


def prepare_features(df):
    """Raw encounter dataframe -> model-ready dataframe (FEATURES columns).

    HIST_COLS are optional in the input (default 0 = no known history).
    """
    df = df.copy()
    for c in HIST_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)
    df["age_num"] = df["age"].map(age_to_num)
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    meds = df[MED_COLS]
    df["n_meds_active"] = (meds != "No").sum(axis=1)
    df["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
    df["inpatient_x_emerg"] = df["number_inpatient"] * (1 + df["number_emergency"])
    df["meds_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    df["labs_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
        df[f"diag_{i}_3d"] = df[f"diag_{i}"].map(icd9_3digit)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[c] = df[c].astype(str)
    X = df[FEATURES].copy()
    X[CAT_COLS] = X[CAT_COLS].fillna("unknown").astype(str)
    return X


def load_data(path=DATA_PATH):
    """Read the raw CSV ('?' = missing), add patient history, drop encounters that cannot be readmitted."""
    df = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    df = add_patient_history(df)
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_EXCLUDE)].reset_index(drop=True)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df
