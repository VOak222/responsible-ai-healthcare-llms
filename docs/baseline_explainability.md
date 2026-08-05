# Baseline Explainability Notes

## Purpose

This step examined which TF-IDF features were most strongly associated with hallucinated and grounded healthcare answers.

## Findings

The features most associated with hallucinated answers included assertive or causal phrases such as "rather than", "significantly", "due to", "enhances", and "increases".

The features most associated with grounded answers included more cautious or scientific wording such as "may", "suggest", "results", and "should".

## Interpretation

The baseline model is useful because it is simple and interpretable. However, the feature importance results show that the model relies heavily on surface-level wording patterns instead of deeper medical reasoning or evidence verification.

This explains why the model can miss fluent hallucinations and sometimes flag technically complex but correct answers as hallucinated.

## Takeaway

The TF-IDF + Logistic Regression baseline is a strong first benchmark, but future models should be evaluated for evidence grounding and semantic understanding, not only text-pattern classification.