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

Your repository must include a `predict.py` that we can run like this:

```bash
python predict.py --input test.jsonl --output scores.jsonl
```

- Each line of `test.jsonl` is `{"id": ..., "text": "..."}`.
- Each line of `scores.jsonl` must be `{"id": ..., "score": <float between 0 and 1>}`.

## Starter code (optional)

This repository contains a simple baseline. You are free to use it, change it, or ignore it.

- **`swedish_quality.jsonl`** contains 100 toy examples, 50 well-written and 50 poorly written. Each has a `text`, a `score` between 0 and 1, a `label` (`good` or `poor`), and, for poor examples, an `error` field describing the mistake. It shows the format but is far too small to be useful as training data on its own.
- **`fit_model.py`** uses the masked-language-modeling head of [AI-Sweden-Models/ModernBERT-large](https://huggingface.co/AI-Sweden-Models/ModernBERT-large). It hides each token in turn and records how likely the model finds the real token in that position. The mean and minimum of these log-probabilities become features for a support vector regressor, trained on an 80/20 split.

To run it:

```bash
pip install torch "transformers>=4.48" scikit-learn
python fit_model.py
```

It runs on a CPU in a few minutes. It prints test-set metrics and, for each test sentence, the token the model found least likely.
