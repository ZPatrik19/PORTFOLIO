# Dry Bean Classification — Project Documentation

## From a notebook-based ML exercise to a reproducible portfolio-grade ML system

This document explains **what was actually built, why each design decision exists, how the project evolved, and how the final repository should be interpreted by a reviewer or recruiter**.

The project solves a seven-class tabular classification problem using the **UCI Dry Bean Dataset**. The original objective was not merely to train a model, but to demonstrate a complete supervised Machine Learning lifecycle:

```text
data acquisition
→ validation
→ exploratory analysis
→ leakage-safe splitting
→ preprocessing
→ classical model benchmarking
→ training-data augmentation experiment
→ PyTorch training
→ hyperparameter comparison
→ final test evaluation
→ error and confidence analysis
→ inference
→ monitoring and drift analysis
→ reproducible reporting
```

The project deliberately treats modeling as **one stage of a larger engineering workflow**.

---

# 1. Problem definition

The dataset contains morphological measurements extracted from dry bean images.

The task is to predict one of seven bean varieties:

```text
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SEKER
SIRA
```

Each observation contains 16 numerical input features describing shape and geometry:

```text
Area
Perimeter
MajorAxisLength
MinorAxisLength
AspectRatio
Eccentricity
ConvexArea
EquivDiameter
Extent
Solidity
Roundness
Compactness
ShapeFactor1
ShapeFactor2
ShapeFactor3
ShapeFactor4
```

This is a multiclass supervised classification problem.

The primary quality metric is **Macro F1**, because each class should contribute equally to the final score. Accuracy is also recorded, but it is not allowed to hide weaker minority-class performance.

---

# 2. Main engineering goal

The repository was designed around one central principle:

> The notebook explains the project, `src/` contains reusable implementation, workflow scripts automate each step, and `run_project.py` connects the complete pipeline.

The final project therefore supports two complementary ways of working.

### Interactive / educational mode

Jupyter notebooks explain:

- what is being done;
- why it is necessary;
- what alternatives exist;
- what intermediate values look like;
- how plots should be interpreted;
- what conclusions follow from each experiment.

### Reproducible / automated mode

The same underlying implementation is available through workflow scripts and a central orchestrator.

The normal user flow is intentionally simple:

```bash
python 00_setup_project.py
python run_project.py
```

The first command prepares the project.

The second executes the complete ML workflow.

---

# 3. Final repository architecture

The repository was reorganized into a top-to-bottom workflow rather than a collection of loosely connected scripts and notebooks.

```text
dry-bean-classification/
│
├── 00_setup_project.py
├── 00_setup_project.ipynb
│
├── 01_data/
│   ├── raw/
│   └── processed/
│
├── workflow/
│   ├── step01_data_acquisition_validation.py
│   ├── step01_data_acquisition_validation.ipynb
│   ├── step02_eda.py
│   ├── step02_eda.ipynb
│   ├── step03_preprocessing.py
│   ├── step03_preprocessing.ipynb
│   ├── step04_baselines.py
│   ├── step04_baselines.ipynb
│   ├── step05_pytorch_training.py
│   ├── step05_pytorch_training.ipynb
│   ├── step06_evaluation_error_analysis.py
│   ├── step06_evaluation_error_analysis.ipynb
│   ├── step07_inference.py
│   ├── step07_inference.ipynb
│   ├── step08_monitoring_drift.py
│   ├── step08_monitoring_drift.ipynb
│   └── step09_generate_report.py
│
├── src/
│   └── dry_bean/
│       ├── augmentation.py
│       ├── benchmark.py
│       ├── config.py
│       ├── constants.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── models.py
│       ├── monitoring.py
│       ├── preprocessing.py
│       ├── training.py
│       ├── utils.py
│       ├── validation.py
│       └── visualization.py
│
├── 03_tests/
├── 04_results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── preexecuted_preview/
│
├── 05_graphviz/
├── tools/
├── reports/
│
├── config.yaml
├── predict.py
├── pyproject.toml
├── requirements.txt
├── run_project.py
├── switch_language.py
└── README.md
```

The structure intentionally avoids excessive nesting. This is a portfolio project, not an enterprise framework.

---

# 4. Execution workflow

The final pipeline is explicitly ordered.

```text
STEP 00 — Project Setup
        ↓
STEP 01 — Data Acquisition & Validation
        ↓
STEP 02 — Exploratory Data Analysis
        ↓
STEP 03 — Split, Preprocessing & Augmentation
        ↓
STEP 04 — Classical Algorithm Benchmark
        ↓
STEP 05 — PyTorch Training & Tuning
        ↓
STEP 06 — Final Evaluation & Error Analysis
        ↓
STEP 07 — Inference
        ↓
STEP 08 — Monitoring & Drift
        ↓
STEP 09 — Report / Figure Generation
```

This removes ambiguity about what should run first and which outputs are required by later stages.

---

# 5. STEP 00 — Automatic project setup

`00_setup_project.py` was created so a new developer does not have to manually reconstruct the Python environment.

The script is responsible for:

```text
create .venv
→ install dependencies
→ install the package with pip install -e .
→ create runtime folders
→ register a Jupyter kernel
→ perform dependency/import smoke tests
→ check the dataset
→ download UCI data when missing
```

Normal setup:

```bash
python 00_setup_project.py
```

Rebuild the environment:

```bash
python 00_setup_project.py --recreate
```

Force a fresh dataset download:

```bash
python 00_setup_project.py --force-data
```

The setup and execution responsibilities are intentionally separated:

```text
00_setup_project.py = prepare the machine/project
run_project.py      = execute the ML pipeline
```

---

# 6. STEP 01 — Data acquisition and schema validation

The official dataset source is UCI Machine Learning Repository dataset ID 602.

The pipeline does not assume that any CSV found on disk is correct.

A strict data contract checks:

- expected 16 features;
- target-column presence;
- numeric feature types;
- missing values;
- duplicate rows;
- known class categories;
- unexpected columns;
- dataset dimensions.

This became especially important during development because even a small feature-name mismatch such as:

```text
AspectRation  vs  AspectRatio
roundness     vs  Roundness
```

can break or silently corrupt a preprocessing pipeline.

The project therefore follows a **fail-fast validation strategy**.

Invalid schema means the pipeline stops before training.

---

# 7. STEP 02 — Exploratory data analysis

EDA was expanded beyond a single class-count chart.

The final project examines:

- class distribution;
- numerical feature distributions;
- correlation structure;
- feature-level outliers;
- class-wise boxplots;
- selected feature-pair scatter plots;
- PCA 2D projection;
- descriptive statistics.

Representative outputs include:

```text
01_class_distribution.png
02_feature_distributions.png
03_correlation_heatmap.png
04_pca_2d.png
eda_boxplots_by_class.png
eda_scatter_1_Area_Perimeter.png
eda_scatter_2_Compactness_Eccentricity.png
```

EDA is not decorative visualization. It is used to answer practical questions such as:

- Are classes balanced?
- Which features are strongly correlated?
- Are some classes geometrically easier to separate?
- Are feature scales very different?
- Are there obvious distributional anomalies?
- Is a nonlinear decision boundary likely to be useful?

---

# 8. STEP 03 — Split and leakage-safe preprocessing

The dataset is separated into:

```text
70% train
15% validation
15% test
```

with stratification.

The roles are intentionally different.

### Training split

Used to fit preprocessing and model parameters.

### Validation split

Used for:

- model comparison;
- hyperparameter decisions;
- early stopping;
- experimental decisions.

### Test split

Reserved for final generalization reporting.

The test split is not used to tune model architecture or hyperparameters.

## StandardScaler leakage protection

The scaler is fit only on training data:

```text
fit(train)
transform(train)
transform(validation)
transform(test)
```

This is a core leakage-control decision.

---

# 9. Training-data augmentation experiment

The official UCI dataset is never overwritten or falsely presented as larger than it is.

Instead, the project contains a clearly separated **training-only SMOTE experiment**.

The logic is:

```text
official dataset
→ train/validation/test split
→ fit StandardScaler on original training data
→ transform training data
→ SMOTE on transformed training data only
→ inverse-transform synthetic rows
→ save train_augmented.csv separately
```

The default configuration uses:

```yaml
augmentation:
  enabled: true
  method: smote
  target_multiplier: 1.15
  k_neighbors: 5
```

In the bundled pre-executed preview:

```text
original training rows : 9,527
augmented training rows: 19,985
synthetic rows added   : 10,458
```

This is explicitly documented as **synthetic interpolation**, not newly collected real observations.

Validation and test data are never augmented.

The project also runs an augmentation ablation:

```text
same model + original train
vs.
same model + SMOTE-augmented train
```

for representative algorithms.

---

# 10. STEP 04 — Heterogeneous model benchmark

The project was expanded into a true model-family benchmark so the neural network is not assumed to be the best choice for tabular data.

The benchmark includes:

| Model family | Algorithm |
|---|---|
| Trivial reference | Dummy Most Frequent |
| Linear | Logistic Regression |
| Dimensionality reduction + linear | PCA + Logistic Regression |
| Generative linear | Linear Discriminant Analysis |
| Probabilistic | Gaussian Naive Bayes |
| Instance based | K-Nearest Neighbors |
| Kernel | RBF Support Vector Machine |
| Single tree | Decision Tree |
| Bagging | Random Forest |
| Randomized trees | Extra Trees |
| Boosting | HistGradientBoosting |
| Boosting | XGBoost |
| Boosting | LightGBM |
| Boosting | CatBoost |
| Ensemble | Soft Voting |

The PyTorch MLP is evaluated alongside these models later.

The objective is to compare **different inductive biases** rather than assume one model family will win.

---

# 11. Cross-validation and model selection

The classical benchmark combines:

```text
training-only stratified cross-validation
+
separate validation holdout
```

Cross-validation records:

```text
CV Macro F1 mean
CV Macro F1 standard deviation
```

This provides both:

```text
quality
+
stability
```

The test set remains untouched until STEP 06.

---

# 12. Metrics

The benchmark records, where supported:

- Accuracy;
- Balanced Accuracy;
- Macro Precision;
- Macro Recall;
- Macro F1;
- Weighted F1;
- Matthews Correlation Coefficient;
- Top-2 Accuracy;
- Multiclass Log Loss;
- One-vs-Rest Macro ROC-AUC;
- Cross-validation Macro F1 mean;
- Cross-validation Macro F1 standard deviation;
- Fit time;
- Inference latency per sample.

This allows model selection to consider both prediction quality and operational characteristics.

---

# 13. STEP 05 — PyTorch neural network

The neural model is a compact fully connected MLP.

Typical architecture:

```text
16 inputs
→ 64 hidden units
→ 32 hidden units
→ 7 logits
```

The training stack uses:

```text
PyTorch Dataset / DataLoader
CrossEntropyLoss
Adam
ReLU
Dropout
early stopping
best-checkpoint restoration
```

The project records per-epoch:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy.

The network is intentionally small because the main purpose is to demonstrate a correct supervised-learning workflow on tabular data.

---

# 14. Hyperparameter benchmark

Multiple MLP configurations are compared.

The tuning space includes:

```text
hidden layer dimensions
dropout
learning rate
batch size
```

The visualization shows readable configuration information and exact Macro F1 values, making the tuning result understandable without opening the underlying CSV.

---

# 15. STEP 06 — Final evaluation

After development decisions are complete, candidate models are evaluated on the untouched test split.

The final evaluation contains:

- all-model comparison;
- normalized confusion matrix;
- per-class precision / recall / F1;
- misclassified samples;
- prediction confidence;
- Top-2 behavior;
- model comparison visualization;
- permutation feature importance.

The workflow therefore distinguishes:

```text
model selection
from
final reporting
```

---

# 16. Error analysis

Incorrect predictions are inspected using:

```text
actual class
predicted class
maximum predicted probability
top-2 margin
```

This supports questions such as:

- Which class pairs are frequently confused?
- Are incorrect predictions generally uncertain?
- Are there high-confidence mistakes?
- Which observations are appropriate for manual review?

---

# 17. Prediction-confidence analysis

Correct and incorrect prediction groups are normalized separately to percentages.

The plot displays:

- sample counts;
- median confidence of correct predictions;
- median confidence of incorrect predictions.

One important conclusion is preserved:

> Softmax confidence is useful as an uncertainty signal, but it is not automatically a calibrated probability of correctness.

---

# 18. Selective classification / manual review

The project evaluates a confidence-threshold policy:

```text
if confidence >= threshold:
    accept automatically
else:
    send to manual review
```

For each threshold it records:

```text
coverage
accepted-sample accuracy
manual-review count
selective error rate
```

The expected trade-off is:

```text
threshold ↑
→ coverage ↓
→ accepted-sample accuracy ↑
```

---

# 19. Model-agnostic feature importance

Permutation importance is computed for the strongest classical candidate.

The logic is:

```text
permute one feature
→ measure score degradation
→ larger degradation = more predictive importance
```

This provides a model-agnostic interpretation layer.

---

# 20. STEP 07 — Inference pipeline

Example:

```bash
python predict.py --input example.json
```

Inference follows:

```text
raw JSON
→ schema validation
→ saved StandardScaler
→ saved label encoder
→ saved PyTorch model
→ logits
→ softmax probabilities
→ predicted class + confidence
```

The inference notebook also discusses:

- invalid input;
- missing features;
- wrong feature types;
- mini-batch inference;
- manual-review thresholds.

---

# 21. STEP 08 — Monitoring and drift simulation

The project simulates production covariate drift by shifting selected feature distributions.

Monitoring includes:

```text
Population Stability Index (PSI)
Kolmogorov-Smirnov statistic
KS p-value
feature-level drift severity
```

The response policy is:

```text
drift detected
→ data-quality check
→ slice analysis
→ labelled performance audit
→ retraining decision
```

---

# 22. Visualization refactor

Reusable visualization logic lives in:

```text
src/dry_bean/visualization.py
```

The same plotting function can be used in notebooks and automated reporting.

Important plots include:

```text
class distribution
feature distributions
correlation matrix
PCA
training loss
training accuracy
hyperparameter comparison
normalized confusion matrix
per-class F1
prediction confidence
model comparison
data drift
classical benchmark
CV stability
multi-metric comparison
performance vs latency
permutation importance
augmentation balance
augmentation ablation
selective classification
```

---

# 23. Notebook philosophy

The notebooks were deliberately **not** reduced to wrapper calls.

Instead, notebooks remain educational and contain:

- objectives;
- theory;
- actual code calls;
- intermediate values;
- tables;
- visualizations;
- interpretation;
- engineering decisions;
- conclusions;
- next-step guidance.

Reusable implementation details remain in `src/`.

---

# 24. HU / EN notebook documentation

The notebook system supports language switching without maintaining two separate notebook sets.

Use:

```bash
python switch_language.py --language en
python switch_language.py --language hu
python switch_language.py --status
```

The switch preserves:

- code cells;
- execution counts;
- DataFrame outputs;
- plots;
- embedded PNGs;
- model outputs.

---

# 25. Testing strategy

The project contains targeted tests under:

```text
03_tests/
```

Coverage includes:

- schema validation;
- configuration loading;
- preprocessing;
- augmentation;
- benchmark contracts;
- PyTorch model input/output;
- inference;
- selective classification;
- monitoring logic;
- runtime-environment behavior.

---

# 26. Reproducibility

Configuration is centralized in:

```text
config.yaml
```

It contains:

- random seed;
- data paths;
- split ratios;
- augmentation settings;
- benchmark settings;
- PyTorch training parameters;
- drift settings;
- result paths;
- artifact paths.

Hard-coded absolute Windows paths were removed from reusable code.

---

# 27. Result and artifact organization

Generated outputs are separated from implementation:

```text
04_results/
├── figures/
├── metrics/
├── models/
└── predictions/
```

Examples:

```text
model_bundle.pt
best_classical_model.joblib
scaler.joblib
label_encoder.joblib

baseline_results.csv
model_comparison.csv
hyperparameter_results.csv
test_metrics.json
drift_report.csv

test_predictions.csv
misclassified_samples.csv
example_prediction.json
```

---

# 28. Pre-executed portfolio preview

The distributed portfolio ZIP includes pre-executed notebook outputs and figures so a reviewer can inspect the project without waiting for all training steps.

The bundled preview was generated on a deterministic schema-compatible surrogate because the build environment did not have access to the UCI download endpoint during rendering.

Therefore:

```text
04_results/preexecuted_preview/
```

contains presentation/demo results.

Those values must **not** be presented as official UCI benchmark results.

The normal runtime pipeline still uses the official UCI dataset.

---

# 29. Two-command user experience

First-time setup:

```bash
python 00_setup_project.py
```

Run everything:

```bash
python run_project.py
```

On Windows, if the global `python` command is not configured:

```powershell
.\.venv\Scripts\python.exe run_project.py
```

---

# 30. What the project demonstrates

The main message is not:

> “I trained a neural network.”

The stronger message is:

> “I can design, validate, compare, evaluate, operationalize and monitor a complete multiclass Machine Learning workflow.”

The project demonstrates:

```text
data contracts
reproducible environments
EDA
leakage prevention
class imbalance handling
synthetic augmentation experiments
model-family benchmarking
cross-validation
holdout model selection
PyTorch training
hyperparameter experimentation
multi-metric evaluation
error analysis
uncertainty-aware decision policies
inference contracts
model-agnostic interpretation
drift detection
testing
reproducible reporting
```

---

# 31. Limitations

### SMOTE does not create real observations

Synthetic rows are interpolations and may not correspond to physically observed beans.

### Probability calibration is not fully implemented

Possible extensions include:

```text
temperature scaling
Platt scaling
isotonic calibration
Expected Calibration Error
reliability diagram
```

### Drift simulation is synthetic

Real monitoring would require production observations over time and ideally delayed ground-truth labels.

### Dataset scope is limited

A real production system would require investigation of acquisition conditions, batch effects, domain shifts, data lineage and real-world class priors.

---

# 32. Future improvements

Strong next extensions would include:

```text
probability calibration
SHAP analysis
Optuna / Bayesian tuning
repeated stratified CV
experiment tracking
MLflow
batch inference API
FastAPI model serving
Docker packaging
CI test workflow
model/version registry
real monitoring dashboard
```

---

# 33. Recommended reviewer path

A recruiter or ML Engineer can understand the project quickly in this order:

```text
1. README.md
2. workflow diagram
3. STEP 02 EDA notebook
4. STEP 03 preprocessing + SMOTE methodology
5. STEP 04 algorithm benchmark
6. STEP 05 PyTorch training
7. STEP 06 evaluation/error analysis
8. STEP 07 inference
9. STEP 08 monitoring
```

For deeper methodology:

```text
reports/PROJECT_METHODOLOGY.md
reports/ALGORITHM_BENCHMARK.md
reports/DATA_AUGMENTATION_METHODOLOGY.md
reports/TECHNICAL_WALKTHROUGH.md
```

---

# 34. Final takeaway

The final repository represents a complete evolution from a standard classification exercise into a structured Machine Learning engineering project.

The core design can be summarized as:

```text
Notebook
= explanation and analysis

src/
= reusable implementation

workflow/*.py
= automated individual steps

run_project.py
= end-to-end orchestration

04_results/
= generated artifacts

03_tests/
= regression protection
```

The project intentionally balances:

```text
simplicity
modularity
reproducibility
education
engineering quality
portfolio readability
```

without turning a medium-sized ML project into an unnecessarily complex framework.

---

## Short portfolio summary

**Dry Bean Classification — Production-style Multiclass ML System**

Built an end-to-end multiclass ML pipeline on the UCI Dry Bean dataset with strict schema validation, leakage-safe preprocessing, classical model-family benchmarking, training-only SMOTE augmentation, PyTorch MLP tuning, multi-metric evaluation, error/confidence analysis, selective classification, JSON inference, drift monitoring, reusable visualizations, automated setup, tests and one-command pipeline execution.

Primary technologies:

```text
Python
Pandas
scikit-learn
PyTorch
imbalanced-learn
XGBoost
LightGBM
CatBoost
Matplotlib
PyYAML
pytest
Jupyter
Graphviz
```
