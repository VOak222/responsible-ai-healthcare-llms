# Baseline Hallucination Detection Findings

## Objective

The goal of this step was to build a first baseline model for detecting hallucinated healthcare answers using the processed MedHallu dataset.

## Dataset Used

The processed dataset contains 20,000 rows created from MedHallu:

- 10,000 grounded/correct answers
- 10,000 hallucinated answers

Each model input combines:

- Question
- Supporting medical knowledge
- Answer

The target label is:

- 0 = Not hallucinated
- 1 = Hallucinated

## Baseline Model

The baseline model uses TF-IDF with Logistic Regression.

TF-IDF converts the text into numerical word and phrase features. Logistic Regression then learns patterns that are associated with hallucinated and non-hallucinated answers.

This baseline is useful because it is simple, fast, interpretable, and gives us a starting point before testing more advanced models.

## Results

The baseline achieved:

- Accuracy: 0.764
- F1 Score: 0.7633

This means the model correctly classified hallucinated vs non-hallucinated answers around 76 percent of the time.

## Error Analysis

On the test set of 4,000 rows:

- Correct predictions: 3,056
- Wrong predictions: 944
- False positives: 466
- False negatives: 478

The errors are fairly balanced. The model sometimes flags correct answers as hallucinated and also misses some hallucinated answers.

## Category-Level Observations

The model performed strongest on:

- Mechanism and Pathway Misattribution

The weakest category was:

- Incomplete Information

One category showed 100 percent accuracy, but it had only 10 test examples, so that result should not be overclaimed.

## Interpretation

The baseline shows that simple text-based features can detect some hallucination patterns in healthcare answers. However, the model likely relies on surface-level wording instead of deep medical reasoning or evidence grounding.

The next step is to improve evaluation by reviewing incorrect examples and then compare this baseline with stronger approaches.