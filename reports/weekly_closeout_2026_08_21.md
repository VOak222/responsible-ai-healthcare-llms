# Weekly Closeout - Responsible AI Healthcare LLM Project

## Work Completed This Week

This week, the project moved from only hallucination detection into broader responsible-AI evaluation.

Main completed items:

1. Integrated Med-HALT Reasoning FCT
   - Processed 37,638 rows into the project format.
   - Compared TF-IDF baseline with semantic similarity.
   - TF-IDF struggled on deeper answer correctness checking.
   - Semantic similarity performed much better because it directly compared answer meaning against the correct answer.

2. Added semantic Med-HALT FCT evaluation
   - Created a reusable semantic evaluation script.
   - Selected threshold using training data.
   - Tested on 7,537 rows.
   - Achieved about 97.9% test accuracy and strong F1 score.

3. Integrated EquityMedQA
   - Added EquityMedQA as the fairness and health-equity dataset.
   - Created a prompt inventory across 7 EquityMedQA files.
   - Total prompts inventoried: 1,722.
   - Split into single prompts and paired prompts.

4. Built EquityMedQA fairness analysis
   - Tagged prompts by fairness category.
   - Categories included gender/sexuality, race/ethnicity, age, income/access, geography, disability, religion/culture, and general clinical/equity prompts.
   - Created review samples for manual inspection.

5. Built EquityMedQA model evaluation template
   - Created a 60-row evaluation template.
   - Includes 40 single fairness prompts and 20 paired demographic comparison prompts.

6. Ran a small Qwen fairness test
   - Tested Qwen/Qwen2.5-0.5B-Instruct on 8 EquityMedQA prompts.
   - Generated responses successfully.
   - Initial review found 4 low observed risk rows and 4 rows needing review.

## What We Learned

The project now has three evaluation layers:

1. Hallucination detection
   - Checks whether an answer is likely false or unsupported.

2. Evidence grounding
   - Checks whether the answer is supported by the provided medical evidence.

3. Fairness and health-equity review
   - Checks whether model behavior changes unfairly based on demographic identity or sensitive context.

The key learning is that high accuracy alone is not enough. A model may perform well on one dataset but still give weak clinical advice, biased language, or inconsistent responses for different patient groups.

## Current Status

The project is on track.

This week successfully added:
- another major dataset,
- semantic answer verification,
- fairness prompt inventory,
- model evaluation template,
- first local Qwen fairness sample run,
- review evidence for Qwen outputs.

## Remaining Work For Next Week

Next week should focus on:

1. Run Qwen on the full 60-row EquityMedQA evaluation template.
2. Improve the fairness review rubric.
3. Add fairness risk into the trustworthiness score.
4. Start preparing a simple dashboard/frontend view.
5. Continue integrating the next dataset if time allows.

## Notes

Large baseline prediction CSV files were not committed because they are output-heavy and not necessary for the GitHub project history. Scripts and meaningful summary results are committed instead.
