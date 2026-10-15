import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description="Run a submission's predict.py on the test set, time it and score it.")
parser.add_argument("--repo", default=".", help="folder containing the submission's predict.py")
parser.add_argument("--input", default="data/test.jsonl")
parser.add_argument("--labels", default="data/test_labels.jsonl")
parser.add_argument("--output", default="scores.jsonl")
args = parser.parse_args()
inp, out = Path(args.input).resolve(), Path(args.output).resolve()

start = time.perf_counter()
subprocess.run([sys.executable, "predict.py", "--input", str(inp), "--output", str(out)], cwd=args.repo, check=True)
elapsed = time.perf_counter() - start  # the whole run, including model loading

labels = {r["id"]: r for r in map(json.loads, open(args.labels, encoding="utf-8"))}
scores = {r["id"]: r["score"] for r in map(json.loads, open(out, encoding="utf-8"))}
assert scores.keys() == labels.keys(), "predict.py must write exactly one score per input id"
assert all(0 <= s <= 1 for s in scores.values()), "every score must be between 0 and 1"

ids = sorted(labels)
pred = [scores[i] for i in ids]
spearman = spearmanr([labels[i]["score"] for i in ids], pred).correlation
auc = roc_auc_score([labels[i]["label"] == "good" for i in ids], pred)

print(f"\n{len(ids)} test examples, timing covers the whole predict.py run including model loading\n")
print("| Spearman | ROC-AUC | Time (s) | Samples/min |")
print("|---|---|---|---|")
print(f"| {spearman:.3f} | {auc:.3f} | {elapsed:.1f} | {len(ids) / elapsed * 60:.0f} |")
