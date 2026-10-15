import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

MODEL = "AI-Sweden-Models/ModernBERT-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

rows = [json.loads(line) for line in open("swedish_quality.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()


class TextDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]["text"]


def collate(texts):
    return tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")


@torch.no_grad()
def embed(rows, batch_size=16):
    out = []
    for enc in DataLoader(TextDataset(rows), batch_size=batch_size, collate_fn=collate):
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())  # mean over real tokens only
    return torch.cat(out).numpy()


X = embed(rows)
y = np.array([r["score"] for r in rows])
labels = [r["label"] for r in rows]

tr, te = train_test_split(np.arange(len(rows)), test_size=0.2, stratify=labels, random_state=0)
reg = make_pipeline(StandardScaler(), SVR(C=1.0)).fit(X[tr], y[tr])
pred = reg.predict(X[te])

print(f"MAE {mean_absolute_error(y[te], pred):.3f} | Spearman {spearmanr(y[te], pred).correlation:.3f}")
print(f"Accuracy (good if > 0.65): {((pred > 0.65) == (y[te] > 0.65)).mean():.2f}\n")
for i, p in sorted(zip(te, pred), key=lambda t: -t[1]):
    print(f"pred {p:.2f}  true {y[i]:.2f}  {rows[i]['text']}")
