# Qwen2.5-1.5B PubMedQA LoRA Fine-Tuning Report

## Objective
Fine-tune Qwen2.5-1.5B-Instruct on PubMedQA using QLoRA/LoRA and compare it fairly with the unchanged base model.

## Compute Environment
- Platform: Google Colab
- GPU: Tesla T4, 14.6 GB VRAM
- Fine-tuning method: QLoRA with LoRA adapters
- Trainable parameters: 18,464,768 (1.182% of total parameters)

## Training Protocol

### Stage 1: Biomedical adaptation
- Dataset: 5,000 PubMedQA artificial examples
- Epochs: 1
- Runtime: 3,775.83 seconds (62.9 minutes)
- Final training loss: 1.4257

### Stage 2: Expert calibration
- Dataset: PubMedQA expert-labelled data
- Labels: yes, no, maybe
- Training rows: 732 after transparent oversampling of the smaller maybe class
- Epochs: 3
- Runtime: 1,756.28 seconds (29.3 minutes)
- Final training loss: 1.4928
- Best observed validation loss: 1.5422 at step 75

## Untouched 200-Row Expert Test Results

| Model | Accuracy | Macro-F1 | Yes Recall | No Recall | Maybe Recall |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-1.5B | 0.555 | 0.248 | 1.000 | 0.015 | 0.000 |
| Qwen2.5-1.5B + PubMedQA LoRA | 0.735 | 0.602 | 0.845 | 0.721 | 0.227 |

## Interpretation
The LoRA adapter improved accuracy by 18 percentage points and macro-F1 by 0.354 on the same untouched expert test set. The base model was heavily biased toward predicting yes. Fine-tuning substantially improved recognition of no decisions and began recognising maybe decisions. The maybe class remains the weakest class and should be monitored in future experiments.

## Next Steps
1. Run the same PubMedQA LoRA protocol for Qwen2.5-3B-Instruct.
2. Rerun Med-HALT using the fine-tuned adapter to check for safety regression or improvement.
3. Evaluate fairness using EquityMedQA paired examples.
