# Diabetes Hospital Readmission Prediction

Predict whether a diabetic patient will be **readmitted to hospital within 30 days** of discharge,
using the UCI *Diabetes 130-US Hospitals (1999-2008)* dataset.

## 1. Problem description

Unplanned readmissions are costly and are often a sign that the discharge plan or follow-up was
insufficient; in many health systems hospitals are financially penalised for them. Diabetic patients
are a high-risk group. Care teams, however, have limited capacity: they cannot phone, visit or
schedule extra appointments for every discharged patient.

**How the model is used:** at discharge time, the hospital sends the encounter data (demographics,
diagnoses, lab flags, medications, number of prior visits...) to the prediction service. The service
returns a probability of readmission within 30 days and a `high_risk` flag. The flag marks the
~20% highest-risk patients, who are prioritised for follow-up calls, medication reviews or an early
outpatient appointment.

* Target: `readmitted == "<30"` (1) vs. everything else (0). About 11% positives (imbalanced).
* Metrics: ROC-AUC (main), PR-AUC, and precision / recall / lift for the flagged top-k% patients.
  Accuracy is not used because 89% of patients are not readmitted.
* This is a hard, noisy problem: published models on this dataset reach ROC-AUC around 0.65-0.70.

## 2. Dataset

[UCI Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/dataset/296):
101,766 inpatient encounters, 50 columns.
`data/diabetic_data.csv` and `data/IDS_mapping.csv` are committed in this repo. To download again:

```bash
curl -L -o data/dataset.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip data/dataset.zip -d data/
```

## 3. Results

Patient-grouped 60/20/20 train/validation/test split (one patient never appears in two splits).
Hyper-parameters tuned on the validation set with random search; test set used once.

| Model | Tuned parameters | Val ROC-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|
| Baseline (prior) | - | 0.500 | - | - |
| Logistic regression | C, class_weight | 0.677 | 0.679 | 0.233 |
| Random forest | trees, depth, leaf size, max_features, class_weight | 0.682 | 0.685 | 0.234 |
| **XGBoost** | trees, lr, depth, min_child_weight, subsample, colsample, lambda, scale_pos_weight | **0.688** | **0.691** | **0.253** |

(`train.py` reports 0.695 test ROC-AUC for the same configuration; differences of about 0.004 come from
row-order randomness in the tree sampling.)

**Feature engineering that helped most:** *patient-history features* (number of earlier encounters,
earlier early-readmissions, whether the previous stay was an early readmission, days spent in
hospital before). They use only a patient's **earlier** encounters, never later ones, and raised validation
ROC-AUC from 0.674 to 0.682 on their own. Finer 3-digit diagnosis codes, ratio features and dropping
very rare drugs added the rest.

Flagging the top 20% highest-risk patients catches ~43% of all early readmissions
(precision ~24% vs. 11% base rate, i.e. 2.1x lift).

## 4. Project structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | Data cleaning, EDA, feature importance, model selection and tuning |
| `features.py` | Data loading + feature engineering shared by notebook, training and serving |
| `train.py` | Trains the final XGBoost model and writes `model.bin` |
| `predict.py` | Flask web service (`/ping`, `/predict`) |
| `test_request.py` | Sends a sample patient to the service |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container for the service |
| `model.bin` | Trained model (committed so the service works out of the box) |

## 5. How to run

### 5.1 Install dependencies (virtual environment)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Notebook

```bash
jupyter notebook notebook.ipynb
```

(Start Jupyter with the venv activated so the notebook uses the right packages.)

### 5.3 Train the model

```bash
python train.py
```

Prints validation / test metrics and saves `model.bin`.

### 5.4 Run the web service locally

```bash
python predict.py                                   # dev server on :9696
# or, production style
gunicorn --bind 0.0.0.0:9696 predict:app            # Linux/macOS
```

Test it:

```bash
curl http://localhost:9696/ping
python test_request.py                              # sends a sample patient
```

Example response:

```json
{"high_risk": true, "readmission_30d_probability": 0.2491, "threshold": 0.1538}
```

`POST /predict` expects a JSON object with the raw encounter fields (same names as in
`diabetic_data.csv`); see `test_request.py` for a complete example. Missing required fields return
HTTP 400 with the list of missing fields.

Optional **patient-history fields** (`n_prior_enc`, `prior_lt30_cnt`, `prior_any_cnt`, `prev_lt30`,
`prev_any`, `prior_days`) can be sent when the hospital knows the patient's earlier stays; if omitted they
default to 0 (no known history) and the score is less accurate.

### 5.5 Docker

```bash
docker build -t diabetes-readmission .
docker run -it --rm -p 9696:9696 diabetes-readmission
python test_request.py            # in another terminal
```

## 6. Cloud deployment (Render)

The service is a Docker image, deployed on [Render](https://render.com) (free web service) straight
from this GitHub repository. `render.yaml` describes the service and the `Dockerfile` binds to the
`$PORT` that Render provides.

1. Push this repo to GitHub.
2. In Render: **New > Blueprint**, pick the repo; Render reads `render.yaml`, builds the Dockerfile and deploys.
3. Test the live service (the free tier sleeps when idle, the first request can take ~1 minute):

```bash
curl https://<your-service>.onrender.com/ping
python test_request.py https://<your-service>.onrender.com
```

**Live service:** https://diabetes-readmission-uii0.onrender.com (free tier: first request after idle may take ~1 minute)

```bash
python test_request.py https://diabetes-readmission-uii0.onrender.com
# {'high_risk': True, 'readmission_30d_probability': 0.2491, 'threshold': 0.1538}
```

## 7. Notes on model performance

ROC-AUC of ~0.69 looks modest but is in line with published work on this dataset: 30-day readmission
depends on factors that are not in the data (social situation, adherence, post-discharge care, lab values).
The dataset is anonymised (no hospital, date or location), so it cannot be joined with other sources.
Ensembling XGBoost with the random forest and logistic regression added only +0.0005 validation AUC,
so the simpler single XGBoost model was kept. The model is intended for **ranking** patients by risk
(2.1x lift in the top 20%), not as a yes/no oracle.

**Live service note:** the deployed Render service is rebuilt from the `main` branch; the service shows the
model that was last pushed.
