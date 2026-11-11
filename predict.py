"""Flask web service that serves the trained readmission model.

Run locally:   python predict.py            (dev server, port 9696)
Production:    gunicorn --bind 0.0.0.0:9696 predict:app
"""
import pickle

import pandas as pd
from flask import Flask, jsonify, request

from features import prepare_features

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]            # sklearn Pipeline (preprocessing + classifier)
threshold = artifact["threshold"]    # probability above which a patient is flagged "high risk"

# Raw fields the client must send (same names as in diabetic_data.csv)
REQUIRED = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
# Fields that may legitimately be missing in the source data ('?') -> optional
OPTIONAL = {"race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"}

app = Flask("readmission")


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json(silent=True)
    if not isinstance(patient, dict):
        return jsonify({"error": "send a JSON object with the encounter fields"}), 400

    missing = [c for c in REQUIRED if c not in patient and c not in OPTIONAL]
    if missing:
        return jsonify({"error": "missing fields", "fields": missing}), 400

    row = {c: patient.get(c) for c in REQUIRED}
    try:
        X = prepare_features(pd.DataFrame([row]))
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:  # bad value types, unknown age bracket, ...
        return jsonify({"error": f"could not score the encounter: {e}"}), 400

    return jsonify({
        "readmission_30d_probability": round(proba, 4),
        "high_risk": proba >= threshold,
        "threshold": round(threshold, 4),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
