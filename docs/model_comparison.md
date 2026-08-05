# Baseline Model Comparison

## Models Compared

This step compared two baseline approaches:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7640 | 0.7633 |
| Sentence Transformer Embeddings + Logistic Regression | 0.5988 | 0.6067 |

## Findings

The TF-IDF baseline performed better than the sentence-transformer embedding baseline.

The sentence-transformer model achieved 59.88 percent accuracy and 60.67 percent F1 score. It detected both hallucinated and non-hallucinated answers, but it produced more false positives and false negatives than the TF-IDF model.

## Interpretation

The sentence-transformer baseline captures semantic similarity, but this task requires more than general sentence meaning. Healthcare hallucination detection depends on whether the answer is grounded in the provided medical knowledge.

The stronger TF-IDF result suggests that surface-level dataset patterns are still very useful for this benchmark. However, both models have limitations and should not be treated as reliable healthcare safety systems.

## Current Best Baseline

The current best baseline is:

TF-IDF + Logistic Regression

with:

- Accuracy: 0.7640
- F1 Score: 0.7633