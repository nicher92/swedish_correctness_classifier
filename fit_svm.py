import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

SEED = 0
MODEL = "AI-Sweden-Models/ModernBERT-large"
set_seed(SEED)  # seeds python, numpy and torch
device = "cuda" if torch.cuda.is_available() else "cpu"

rows = [json.loads(line) for line in open("swedish_quality.jsonl", encoding="utf-8")]
texts = [r["text"] for r in rows]
y = np.array([r["score"] for r in rows])
is_good = np.array([r["label"] == "good" for r in rows])
groups = [r["group"] for r in rows]  # a sentence and its corrupted twin stay in the same fold
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(texts, is_good, groups))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()


@torch.no_grad()
def embed(texts, batch_size=16):
    collate = lambda b: tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
    out = []
    for enc in DataLoader(texts, batch_size=batch_size, collate_fn=collate):
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())  # mean over real tokens only
    return torch.cat(out).numpy()


start = time.perf_counter()
X = embed(texts)
pred = cross_val_predict(make_pipeline(StandardScaler(), SVR(C=1.0)), X, y, cv=folds)
elapsed = time.perf_counter() - start

print(f"\n{len(texts)} sentences, 5-fold stratified group CV, seed {SEED}, device {device}\n")
print("| Method | Spearman | ROC-AUC | Time (s) | Samples/min |")
print("|---|---|---|---|---|")
print(f"| Mean-pooled embeddings + SVR | {spearmanr(y, pred).correlation:.3f} | {roc_auc_score(is_good, pred):.3f}"
      f" | {elapsed:.1f} | {len(texts) / elapsed * 60:.0f} |")
