# EquityMedQA Qwen Full Fairness Evaluation Summary

## What We Completed

Today we moved from the small 8-row Qwen pilot to the full EquityMedQA fairness evaluation template.

- Model used: Qwen/Qwen2.5-0.5B-Instruct
- Total prompts processed: 60
- Single fairness prompts: 40
- Paired demographic comparison prompts: 20
- Manual review shortlist created: 33 rows

## Main Result

The model generated responses for all prompts, so the run was technically successful. However, the fairness and safety screening found that many responses still need manual checking before we can treat them as safe or fair.

| evaluation_type | overall_fairness_safety_risk | rows |
| --- | --- | --- |
| paired_prompt_comparison | low_observed_risk | 8 |
| paired_prompt_comparison | review_needed | 12 |
| single_prompt_fairness | low_observed_risk | 19 |
| single_prompt_fairness | review_needed | 21 |

## Why Rows Were Flagged

| flag | count |
| --- | --- |
| missing_professional_care_guidance | 23 |
| possible_unsafe_clinical_advice | 7 |
| possible_stereotype_language | 3 |
| paired_response_length_difference | 3 |
| pregnancy_medication_caution_needed | 1 |

## Review Needed By Category

| fairness_category | review_needed_rows |
| --- | --- |
| age | 3 |
| disability | 2 |
| gender_sexuality | 4 |
| general_equity_or_clinical_prompt | 5 |
| geography | 1 |
| income_access | 1 |
| race_ethnicity | 1 |
| religion_culture | 4 |
| paired_prompt_comparison | 12 |

## Interpretation

The Qwen model usually writes polite and complete answers, but that is not enough for healthcare evaluation. Some responses gave clinical advice too confidently, some missed clear professional-care guidance, and some paired prompts changed too much when demographic wording changed.

This matters because fairness evaluation is not only about whether the answer sounds respectful. We also need to check whether the answer stays clinically safe, avoids stereotypes, and treats similar patients consistently.

## Main Takeaway

EquityMedQA is now integrated as the fairness and health-equity evaluation layer of the project. The project can now evaluate not only hallucination and groundedness, but also whether model responses may become unsafe or inconsistent when demographic context is added.

## Next Step

The next step is manual review of the 33 shortlisted rows. After that, we can convert the review labels into a fairness risk score and connect that score into the broader Clinical Trustworthiness Score.
