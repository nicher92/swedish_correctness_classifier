import json
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from transformers import set_seed
from predict import load_encoder, embed

SEED = 0
set_seed(SEED)  # seeds python, numpy and torch

rows = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8")]
tok, model = load_encoder()
X = embed([r["text"] for r in rows], tok, model)
y = np.array([r["score"] for r in rows])

regressor = make_pipeline(StandardScaler(), SVR(C=1.0)).fit(X, y)
joblib.dump(regressor, "svr.joblib")
print(f"Trained on {len(rows)} examples and saved svr.joblib")
