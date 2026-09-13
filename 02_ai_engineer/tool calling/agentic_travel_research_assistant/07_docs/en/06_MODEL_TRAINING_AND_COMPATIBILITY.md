# Model training and compatibility

This chapter documents the trainable intent-router pipeline, split strategy, threshold selection and safe model-loading behavior.

## 1. Task

The intent router performs multi-label classification. A single request can activate several labels such as `weather + restaurant + transport`.

## 2. Feature pipeline

```text
word TF-IDF (1–3 grams)
+ Unicode accent normalization
      ↓
One-vs-Rest SGD logistic classifiers
      ↓
per-label probability
      ↓
validation-selected threshold
```

Character n-grams are particularly useful for Hungarian inflection, missing accents and minor spelling noise.

## 3. Splits and thresholds

192k train / 24k validation / 24k held-out test. Thresholds are selected independently per label with a precision-oriented objective because unnecessary tool calls are a real agent failure mode. Current thresholds are stored in the model metrics JSON.

## 4. Current model metrics

- test micro-F1: ~0.783
- test macro-F1: ~0.785
- test hamming loss: ~0.131
- 36k challenge micro-F1: ~0.677

The challenge score is intentionally lower because it uses indirect/noisy language and therefore acts as a stronger generalization stress test.

## 5. Persistence and runtime compatibility

Joblib/scikit-learn artifacts are not treated as runtime-independent. A sidecar `intent_router_metadata.json` stores Python/scikit-learn versions and the dataset hash. Setup checks metadata before unpickling. On runtime mismatch it retrains once locally instead of loading an incompatible estimator.

## 6. When is retraining triggered?

Only when the model is missing, runtime-incompatible, metadata is missing/invalid, or the training-dataset hash changed. Normal startup does not retrain unnecessarily.

## Conclusion

The router is a reproducible, measurable tool-selection component rather than a universal NLP model. Split integrity and runtime compatibility are as important as classifier accuracy.
