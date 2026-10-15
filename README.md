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

## Paper

Write a short paper of about 2 pages, excluding references. A sentence or two per point is often enough. It should cover:

- **Approach:** what you built and why you chose it over the alternatives.
- **Data:** where your training data comes from, how you created or cleaned it, and how you labeled it.
- **Experiments:** what you tried, including what didn't work, and how you evaluated it.
- **Results:** your performance numbers and speed, with the hardware you measured on.
- **Limitations:** the kinds of errors your model misses or misjudges, with examples.

## Starter code (optional)

This repository contains a simple baseline. You are free to use it, change it, or ignore it.

- **`swedish_quality.jsonl`** contains 598 toy examples, 299 well-written and 299 poorly written, covering a range of topics and lengths. Most poor examples are a copy of a good example with one to three errors added. Each row has a `text`, a `score` between 0 and 1, a `label` (`good` or `poor`), an `error` field describing the mistakes in poor examples, and a `group` ID that a good example shares with its corrupted copy. It shows the format but is far too small to be useful as training data on its own.
- **`fit_svm.py`** runs each text through [AI-Sweden-Models/ModernBERT-large](https://huggingface.co/AI-Sweden-Models/ModernBERT-large) and averages the token vectors into one vector per text. A support vector regressor then learns to predict the score from these vectors. It is evaluated with seeded 5-fold stratified group cross-validation, so every example is predicted by a model that never saw it or its corrupted copy. The script prints Spearman correlation, ROC-AUC, run time and samples per minute as a Markdown table.

To run it:

```bash
pip install torch "transformers>=4.48" scikit-learn
python fit_svm.py
```

The first run downloads the model (about 1.6 GB). After that, it runs on a CPU in under a minute.

### Baseline results

Results from `fit_svm.py` on the 598 toy examples, run on a laptop CPU:

| Method | Spearman | ROC-AUC | Time (s) | Samples/min |
|---|---|---|---|---|
| Mean-pooled embeddings + SVR | 0.587 | 0.836 | 19.7 | 1821 |# Swedish Grammaticality Scoring: Master's Thesis Assignment

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

## Paper

Write a short paper of about 2 pages, excluding references. A sentence or two per point is often enough. It should cover:

- **Approach:** what you built and why you chose it over the alternatives.
- **Data:** where your training data comes from, how you created or cleaned it, and how you labeled it.
- **Experiments:** what you tried, including what didn't work, and how you evaluated it.
- **Results:** your performance numbers and speed, with the hardware you measured on.
- **Limitations:** the kinds of errors your model misses or misjudges, with examples.

## Starter code (optional)

This repository contains a simple baseline. You are free to use it, change it, or ignore it.

- **`swedish_quality.jsonl`** contains 598 toy examples, 299 well-written and 299 poorly written, covering a range of topics and lengths. Most poor examples are a copy of a good example with one to three errors added. Each row has a `text`, a `score` between 0 and 1, a `label` (`good` or `poor`), an `error` field describing the mistakes in poor examples, and a `group` ID that a good example shares with its corrupted copy. It shows the format but is far too small to be useful as training data on its own.
- **`fit_svm.py`** runs each text through [AI-Sweden-Models/ModernBERT-large](https://huggingface.co/AI-Sweden-Models/ModernBERT-large) and averages the token vectors into one vector per text. A support vector regressor then learns to predict the score from these vectors. It is evaluated with seeded 5-fold stratified group cross-validation, so every example is predicted by a model that never saw it or its corrupted copy. The script prints Spearman correlation, ROC-AUC, run time and samples per minute as a Markdown table.

To run it:

```bash
pip install torch "transformers>=4.48" scikit-learn
python fit_svm.py
```

The first run downloads the model (about 1.6 GB). After that, it runs on a CPU in under a minute.

### Baseline results

Results from `fit_svm.py` on the 598 toy examples, run on a laptop CPU:

| Method | Spearman | ROC-AUC | Time (s) | Samples/min |
|---|---|---|---|---|
| Mean-pooled embeddings + SVR | 0.587 | 0.836 | 19.7 | 1821 |
