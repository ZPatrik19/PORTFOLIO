# Final Enhancement Summary

This release extends the previous benchmark-oriented version without changing the
core UCI task.

## Added

- training-only SMOTE augmentation pipeline;
- separate `train_augmented.csv` processed artifact;
- explicit original-vs-augmented ablation on Logistic Regression, Random Forest
  and HistGradientBoosting;
- augmentation class-balance visualization;
- selective-classification coverage/accuracy analysis;
- richer inference notebook with contract failures and mini-batch scoring;
- richer monitoring notebook with Stable / Watch / Material Drift actions;
- feature dictionary and descriptive statistics in STEP 01;
- parameter-count explanation in STEP 05;
- dedicated augmentation methodology document;
- detailed technical walkthrough;
- 2 additional real tests covering augmentation and selective classification.

## Data provenance rule

The official UCI raw dataset is never expanded or overwritten. Synthetic rows
exist only as a separate training artifact and are always labelled/documented as
SMOTE-generated data.

## Preview packaging

Executed notebook outputs remain included for immediate portfolio review. The
preview dataset is still the explicitly documented build-time schema-compatible
surrogate, not an official UCI benchmark result.
