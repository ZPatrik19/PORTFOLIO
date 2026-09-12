# Dry Bean Classification — Production-style Multiclass ML System

End-to-end ML portfolio project for classifying seven dry-bean varieties from 16 morphological measurements. The emphasis is the complete, reproducible lifecycle **and a meaningful model-family benchmark**, not neural-network accuracy alone.


## Two-command workflow

The project is designed to be used with two commands:

```bash
python 00_setup_project.py
python run_project.py
```

`00_setup_project.py` creates the isolated `.venv`, installs the required
packages, registers the Jupyter kernel, validates the installation and downloads
the official UCI Dry Bean dataset if it is missing.

`run_project.py` then automatically uses that `.venv` and executes the complete
STEP 01 → STEP 09 workflow in order.

**Manual environment activation is not required.**

See [`reports/QUICK_START.md`](reports/QUICK_START.md) for details.
## Problem

```text
16 raw morphological features
→ validation
→ leakage-safe preprocessing
→ heterogeneous classical benchmark + PyTorch MLP
→ validation/CV model comparison
→ untouched test evaluation
→ error/confidence analysis
→ JSON inference
→ drift monitoring
```

Target classes: `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SIRA`.

## Workflow

![Workflow](04_results/preexecuted_preview/figures/workflow.svg)

```text
00 Setup
↓
01 Data Acquisition & Validation
↓
02 EDA
↓
03 Split & Preprocessing
├─→ 04 Classical Algorithm Benchmark ─┐
└─→ 05 PyTorch Training ──────────────┤
                                      ↓
06 Final Evaluation & Error Analysis
↓
07 Inference
↓
08 Monitoring / Drift
↓
09 Report Figures
```

## Project structure

```text
.
├── 00_setup_project.py
├── 00_setup_project.ipynb
├── 01_data/
│   ├── raw/
│   └── processed/
├── workflow/
│   ├── step01_data_acquisition_validation.{py,ipynb}
│   ├── step02_eda.{py,ipynb}
│   ├── step03_preprocessing.{py,ipynb}
│   ├── step04_baselines.{py,ipynb}
│   ├── step05_pytorch_training.{py,ipynb}
│   ├── step06_evaluation_error_analysis.{py,ipynb}
│   ├── step07_inference.{py,ipynb}
│   ├── step08_monitoring_drift.{py,ipynb}
│   └── step09_generate_report.py
├── src/dry_bean/
├── 03_tests/
├── 04_results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── preexecuted_preview/
├── 05_graphviz/
├── tools/
├── reports/
├── run_project.py
├── predict.py
├── switch_language.py
├── config.yaml
├── requirements.txt
└── pyproject.toml
```

## Quick start

### STEP 00 — automatic setup

```bash
python 00_setup_project.py
```

Recreate the environment:

```bash
python 00_setup_project.py --recreate
```

Force dataset download:

```bash
python 00_setup_project.py --force-data
```

The setup script does not try to activate the parent shell. Activate manually after setup.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Smoke test pipeline

```bash
python run_project.py --quick
```

### Full pipeline

```bash
python run_project.py
```

Selected steps:

```bash
python run_project.py --steps 1-4
python run_project.py --steps 3,5,6
```

## Notebook workflow

Open `00_setup_project.ipynb`, then the notebooks in `workflow/` from STEP 01 to STEP 08. The notebooks explain **what**, **why**, alternatives, intermediate outputs and interpretation; reusable logic remains in `src/dry_bean/`.

The ZIP is intentionally shipped with executed notebook outputs so the analysis and plots are visible immediately. See [`reports/PREEXECUTED_PREVIEW.md`](reports/PREEXECUTED_PREVIEW.md) for the exact provenance of those bundled preview outputs.

Detailed guide: [`reports/NOTEBOOK_GUIDE.md`](reports/NOTEBOOK_GUIDE.md).

## Algorithm benchmark

STEP 04 compares multiple model families rather than treating the neural network as the default winner:

| Family | Model | Why it is included |
|---|---|---|
| Sanity baseline | Dummy Most Frequent | verifies that learned models beat a trivial strategy |
| Linear | Logistic Regression | strong, interpretable linear reference |
| Dimensionality reduction + linear | PCA + Logistic Regression | tests whether a compact orthogonal representation helps |
| Generative linear | Linear Discriminant Analysis | exploits class-separation structure under distributional assumptions |
| Probabilistic | Gaussian Naive Bayes | very fast conditional-independence baseline |
| Instance based | KNN | captures local nonlinear neighborhoods |
| Kernel | RBF SVM | flexible nonlinear decision boundaries on scaled tabular data |
| Single tree | Decision Tree | interpretable nonlinear baseline and overfitting reference |
| Bagging ensemble | Random Forest | robust nonlinear tree ensemble |
| Randomized ensemble | Extra Trees | higher randomization / lower-variance tree ensemble |
| Boosting | HistGradientBoosting | strong scikit-learn boosting baseline |
| Gradient boosting | XGBoost | regularized boosted trees; one of the standard references for tabular ML |
| Gradient boosting | LightGBM | histogram-based boosting optimized for speed and structured data |
| Gradient boosting | CatBoost | robust boosted-tree reference with strong regularization and class weighting |
| Ensemble | Soft Voting | tests whether complementary probability estimates improve robustness |
| Neural network | PyTorch MLP | learned nonlinear representation with early stopping and tuning |

The classical benchmark uses **training-only stratified cross-validation** for stability estimates and a separate validation split for model comparison. The untouched test set is reserved for STEP 06.

XGBoost, LightGBM and CatBoost are included in the full benchmark because they are highly relevant production-grade tabular learners. The benchmark code imports them lazily, so the core sklearn comparison still runs in a constrained environment, while the supplied `requirements.txt` installs them for full reproducibility.

## Performance metrics

Accuracy alone is not sufficient. The benchmark/final evaluation records, where available:

- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- Matthews correlation coefficient (MCC);
- Top-2 accuracy;
- multiclass log loss;
- macro One-vs-Rest ROC-AUC;
- stratified-CV Macro-F1 mean and standard deviation;
- model fit time;
- inference latency per sample.

**Macro F1 remains the primary quality metric** because each bean class contributes equally, while CV standard deviation gives a simple robustness/stability signal.

## Model-comparison visualizations

The report now includes dedicated model-methodology views in addition to EDA/training/error plots:

- classical algorithm ranking;
- cross-validation stability with uncertainty;
- multi-metric final-test heatmap;
- performance-vs-inference-latency trade-off;
- model-agnostic permutation feature importance;
- PyTorch hyperparameter benchmark;
- final all-model comparison;
- training-data augmentation class balance;
- augmentation ablation (original vs SMOTE);
- selective-classification coverage vs accepted-sample accuracy.

Pre-rendered preview examples are available under `04_results/preexecuted_preview/figures/`.


## Training-data augmentation

The official UCI Dry Bean dataset is **never inflated or overwritten**. To study whether additional training volume helps, STEP 03 creates an optional **training-only SMOTE** dataset.

The default recipe:

```text
official raw data
→ deduplication
→ stratified 70/15/15 split
→ fit StandardScaler on ORIGINAL training split only
→ SMOTE in standardized training feature space
→ inverse-transform synthetic rows
→ save separate train_augmented.csv
```

With the default `target_multiplier: 1.15`, each class is grown to roughly 115% of the largest original training class. On the bundled preview this increases the training set from about 9.5k rows to about 20k rows.

This does **not** mean the project suddenly owns twice as many real observations. SMOTE rows are synthetic interpolations. Validation and test remain untouched, and STEP 04 runs an explicit augmentation ablation to determine whether the larger training set actually improves Macro F1.

Detailed methodology: [`reports/DATA_AUGMENTATION_METHODOLOGY.md`](reports/DATA_AUGMENTATION_METHODOLOGY.md).

## PyTorch model

The neural model is a compact MLP, normally starting around `16 → 64 → 32 → 7`, ReLU + dropout, Adam, `CrossEntropyLoss` and validation-loss early stopping. A small hyperparameter matrix compares learning rate, hidden dimensions, dropout and batch size.

The project **does not force the MLP to win**. If a simpler classical model performs as well or better, the conclusion should reflect that result.

## Evaluation and error analysis

STEP 06 evaluates every saved candidate on the untouched test split after model development. It produces an all-model comparison table plus detailed PyTorch diagnostics:

- normalized confusion matrix;
- per-class precision/recall/F1;
- misclassified samples;
- maximum softmax confidence;
- top-2 margin;
- confidence distribution for correct vs incorrect predictions.

It also computes **permutation importance** for the best classical candidate selected on validation, providing model-agnostic feature-level interpretation.

## Inference

After training:

```bash
python predict.py --input example.json
```

Output contract:

```json
{
  "predicted_class": "SEKER",
  "confidence": 0.94,
  "class_probabilities": {"SEKER": 0.94}
}
```

## Monitoring

STEP 08 synthetically shifts selected production features and calculates PSI and two-sample KS statistics. Drift is treated as a warning signal, not automatic proof of model-performance degradation.

## Tests

```bash
pytest -q
```

Tests cover schema validation, preprocessing, model input/output, inference, drift logic and the heterogeneous benchmark contract. The current refactor passes the supplied test suite.

## HU / EN notebook documentation

```bash
python switch_language.py --status
python switch_language.py --language en
python switch_language.py --language hu
```

Language switching updates markdown source only. Code cells, execution counts and outputs are preserved.

## Figures and reusable visualization

Reusable plots live in `src/dry_bean/visualization.py`. Notebooks display them inline; STEP 09 saves the same `Figure` implementations to `04_results/figures/`. This removes notebook-vs-script plotting duplication.

## Pre-executed ZIP preview

The packaged repository includes executed notebook outputs and pre-rendered figures so the project is immediately inspectable without running training first. In the build environment the official UCI dataset could not be downloaded directly, therefore these **bundled preview outputs were generated from a deterministic schema-compatible synthetic surrogate** and are stored separately under:

```text
04_results/preexecuted_preview/
```

They demonstrate the code paths and presentation only and **must not be reported as official UCI benchmark results**. Running the normal project downloads UCI Dry Bean (ID 602) and regenerates standard outputs in `04_results/{figures,metrics,models,predictions}`.

## Detailed methodology

See [`reports/PROJECT_METHODOLOGY.md`](reports/PROJECT_METHODOLOGY.md) for model-selection methodology, leakage rules, metrics, interpretation and limitations.

## Limitations / future work

Useful extensions include probability calibration, coverage-vs-accuracy threshold analysis, repeated-seed evaluation, Optuna/Bayesian optimization, SHAP/feature-attribution analysis, experiment tracking and operationally calibrated drift thresholds.

## Documentation

- [`reports/PROJECT_METHODOLOGY.md`](reports/PROJECT_METHODOLOGY.md) — modeling decisions and evaluation philosophy.
- [`reports/DATA_AUGMENTATION_METHODOLOGY.md`](reports/DATA_AUGMENTATION_METHODOLOGY.md) — why/how the training set is synthetically expanded and what its limitations are.
- [`reports/TECHNICAL_WALKTHROUGH.md`](reports/TECHNICAL_WALKTHROUGH.md) — end-to-end technical walkthrough for reviewers/interviews.
- [`reports/NOTEBOOK_GUIDE.md`](reports/NOTEBOOK_GUIDE.md) — notebook execution order, inputs and outputs.
