# Swedish Grammaticality Scoring: Master's Thesis Assignment

## Task

Build a model that grades the grammatical correctness of Swedish text. Given a text, it should output a single score between 0 and 1, where well-written Swedish scores higher than poorly written Swedish.

You choose the approach: encoders, decoders, encoder-decoders, fine-tuning, prompting, feature extraction, or something else entirely. You also choose your training data. You can use existing datasets, generate your own, or combine both.

## Examples

| Input | Expected score |
|---|---|
| I norröna tid fanns det svält i Oslo och många människor dog inte hade fisk tillgång från England, medan man hoppades att det skulle komma hem när fisken återvände den nästa. Norge fick inte några ny fiskelever som kunde vara framme. | ~0.3 |
| Kommunen har beslutat att bygga en ny förskola i stadsdelen, och arbetet beräknas vara klart till hösten nästa år. | ~0.98 |

## Constraints

- The model must have fewer than 1.3B parameters and be published in Hugging Face format on the Hugging Face Hub.
- Your code must be on GitHub, and your training data must be published on the Hugging Face Hub.
- Your results must be reproducible. Set all random seeds, pin your dependency versions, and document how to reproduce training from scratch.

## Evaluation

Submissions are evaluated on:

- **Performance** on our held-out test set, which is not shared.
- **Speed**, measured in graded samples per minute. All submissions are run with the same setup on our hardware.
- **Code quality**, meaning readability, structure, and how easy the code is to run.

## Submission

Send us:

1. A link to your GitHub repository.
2. Your Hugging Face model ID.
3. Your Hugging Face dataset ID(s).
4. A short paper describing your work (see below).

Your repository must include a `predict.py` that we can run like this:

```bash
python predict.py --input test.jsonl --output scores.jsonl
```

- Each line of `test.jsonl` is `{"id": ..., "text": "..."}`.
- Each line of `scores.jsonl` must be `{"id": ..., "score": <float between 0 and 1>}`.

You can check that your `predict.py` works with our setup by running `evaluate.py` from the starter code (see below).

## Paper

Write a short paper of about 2 pages, excluding references. A sentence or two per point is often enough. It should cover:

- **Approach:** what you built and why you chose it over the alternatives.
- **Data:** where your training data comes from, how you created or cleaned it, and how you labeled it.
- **Experiments:** what you tried, including what didn't work, and how you evaluated it.
- **Results:** your performance numbers and speed, with the hardware you measured on.
- **Limitations:** the kinds of errors your model misses or misjudges, with examples.

## Starter code (optional)

This repository contains a simple baseline. You are free to use it, change it, or ignore it.

**Data.** 598 toy examples covering a range of topics and lengths, split 80/20 into training and test sets. Most poor examples are a copy of a good example with one to three errors added, and both copies are always in the same split. It shows the format but is far too small to be useful as training data on its own.

- `data/train.jsonl` has 478 examples. Each has an `id`, a `text`, a `score` between 0 and 1, a `label` (`good` or `poor`), an `error` field describing the mistakes in poor examples, and a `group` ID that a good example shares with its corrupted copy.
- `data/test.jsonl` has 120 examples with only `id` and `text`, the same format as our hidden test set.
- `data/test_labels.jsonl` has the `score` and `label` for each test `id`.

**Scripts.**

- `train.py` runs each training text through [AI-Sweden-Models/ModernBERT-large](https://huggingface.co/AI-Sweden-Models/ModernBERT-large) and averages the token vectors into one vector per text. A support vector regressor learns to predict the score from these vectors and is saved as `svr.joblib`.
- `predict.py` follows the required submission interface: it reads texts, embeds them the same way, and writes one score between 0 and 1 per text.
- `evaluate.py` runs `predict.py` the same way we will run yours. It times the whole run, including model loading, checks the output format, and prints Spearman correlation, ROC-AUC and samples per minute as a Markdown table. Use `--repo path/to/your/repo` to evaluate a different `predict.py`.

To run them:

```bash
pip install torch "transformers>=4.48" scikit-learn
python train.py
python evaluate.py
```

The first run downloads the model (about 1.6 GB). After that, each script runs on a CPU in under a minute.

### Baseline results

Results from `evaluate.py` on `data/test.jsonl`, run on a laptop CPU:

| Spearman | ROC-AUC | Time (s) | Samples/min |
|---|---|---|---|
| 0.528 | 0.783 | 9.8 | 731 |
