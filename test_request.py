"""Send a sample encounter to the running service (default: local)."""
import sys
import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"

patient = {
    "race": "Caucasian", "gender": "Female", "age": "[70-80)",
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "time_in_hospital": 5, "payer_code": "MC", "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18,
    "number_outpatient": 0, "number_emergency": 2, "number_inpatient": 3,
    "diag_1": "428", "diag_2": "250.01", "diag_3": "401", "number_diagnoses": 9,
    "max_glu_serum": "None", "A1Cresult": ">8", "change": "Ch", "diabetesMed": "Yes",
    "metformin": "No", "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
    "glimepiride": "No", "acetohexamide": "No", "glipizide": "No", "glyburide": "No",
    "tolbutamide": "No", "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
    "miglitol": "No", "troglitazone": "No", "tolazamide": "No", "insulin": "Up",
    "glyburide-metformin": "No", "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    # optional patient-history fields (omit them if the history is unknown)
    "n_prior_enc": 3, "prior_lt30_cnt": 1, "prior_any_cnt": 2,
    "prev_lt30": 1, "prev_any": 1, "prior_days": 12,
}

print(requests.post(f"{url}/predict", json=patient).json())
