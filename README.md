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
| Logistic regression | C, class_weight | 0.666 | 0.671 | 0.216 |
| Random forest | trees, depth, leaf size, max_features, class_weight | 0.671 | 0.678 | 0.225 |
| **XGBoost** | trees, lr, depth, min_child_weight, subsample, colsample, scale_pos_weight | **0.674** | **0.683** | **0.238** |

Flagging the top 20% highest-risk patients catches ~40% of all early readmissions
(precision ~23% vs. 11% base rate, i.e. 2x lift).

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
{"high_risk": true, "readmission_30d_probability": 0.4631, "threshold": 0.3417}
```

`POST /predict` expects a JSON object with the raw encounter fields (same names as in
`diabetic_data.csv`); see `test_request.py` for a complete example. Missing required fields return
HTTP 400 with the list of missing fields.

### 5.5 Docker

```bash
docker build -t diabetes-readmission .
docker run -it --rm -p 9696:9696 diabetes-readmission
python test_request.py            # in another terminal
```

## 6. Cloud deployment

The service is a standard Docker image listening on port 9696, so it can be deployed to any
container platform. Example with [Fly.io](https://fly.io):

```bash
fly auth login
fly launch --name diabetes-readmission --internal-port 9696 --no-deploy
fly deploy
python test_request.py https://diabetes-readmission.fly.dev
```

*Status: not deployed yet. Add the public URL or a screenshot here once deployed.*
