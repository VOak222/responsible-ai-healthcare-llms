# Qwen2.5-3B Med-HALT Smoke Test

## Purpose

This smoke test checks whether Qwen2.5-3B-Instruct can run locally for the healthcare hallucination evaluation pipeline.

This is not a final model result. It is only a 5-row test to check model loading, runtime, output parsing, and basic compatibility.

## Result

| Metric | Value |
|---|---:|
| Rows tested | 5 |
| Parse success | 1.0 |
| Smoke test accuracy | 0.8 |
| Total runtime seconds | 55.62 |

## Interpretation

If parse success is close to 1.0 and runtime is manageable, Qwen2.5-3B can be tested on a larger sample next.

If runtime is too slow, Phi-3.5-mini or a quantized model should be used as the backup candidate.
