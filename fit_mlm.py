import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
model = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()


@torch.no_grad()
def token_logprobs(text):
    enc = tok(text, return_tensors="pt", return_special_tokens_mask=True).to(device)
    ids = enc["input_ids"][0]
    pos = torch.where(enc["special_tokens_mask"][0] == 0)[0]  # real tokens, not [CLS]/[SEP]
    n = torch.arange(len(pos), device=device)
    batch = ids.repeat(len(pos), 1)
    batch[n, pos] = tok.mask_token_id                           # copy i has token i masked
    logits = model(input_ids=batch).logits[n, pos].float()
    logp = logits.log_softmax(-1)[n, ids[pos]]                  # log P(actual token | context)
    return logp.cpu().numpy(), [tok.decode([t]) for t in ids[pos].tolist()]


results = [token_logprobs(r["text"]) for r in rows]
X = np.array([[lp.mean(), lp.min()] for lp, _ in results])
y = np.array([r["score"] for r in rows])
labels = [r["label"] for r in rows]

tr, te = train_test_split(np.arange(len(rows)), test_size=0.2, stratify=labels, random_state=0)
print(f"No training, min log-prob alone: Spearman {spearmanr(y[te], X[te, 1]).correlation:.3f}")

reg = make_pipeline(StandardScaler(), SVR(C=1.0)).fit(X[tr], y[tr])
pred = reg.predict(X[te])
print(f"SVR: MAE {mean_absolute_error(y[te], pred):.3f} | Spearman {spearmanr(y[te], pred).correlation:.3f}"
      f" | Accuracy (good if > 0.65): {((pred > 0.65) == (y[te] > 0.65)).mean():.2f}\n")

for i, p in sorted(zip(te, pred), key=lambda t: -t[1]):
    lp, toks = results[i]
    print(f"pred {p:.2f}  true {y[i]:.2f}  worst {toks[lp.argmin()]!r} ({lp.min():.1f})  {rows[i]['text']}")
