import argparse
import json
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

MODEL = "AI-Sweden-Models/ModernBERT-large"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    return tok, model


@torch.no_grad()
def embed(texts, tok, model, batch_size=16):
    collate = lambda b: tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
    out = []
    for enc in DataLoader(texts, batch_size=batch_size, collate_fn=collate):
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())  # mean over real tokens only
    return torch.cat(out).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Swedish texts for grammatical correctness (0 = poor, 1 = good).")
    parser.add_argument("--input", required=True, help="JSONL with one {'id', 'text'} per line")
    parser.add_argument("--output", required=True, help="JSONL to write one {'id', 'score'} per line")
    parser.add_argument("--model", default="svr.joblib", help="regressor saved by train.py")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    tok, model = load_encoder()
    regressor = joblib.load(args.model)
    scores = np.clip(regressor.predict(embed([r["text"] for r in rows], tok, model)), 0, 1)  # SVR can overshoot

    with open(args.output, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "score": float(s)}) + "\n")
