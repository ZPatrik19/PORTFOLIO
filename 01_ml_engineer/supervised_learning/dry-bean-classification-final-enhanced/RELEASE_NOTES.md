# Release Notes — Extended Algorithm Benchmark

This ZIP contains the upgraded Dry Bean ML portfolio project.

Key additions:
- 15 classical/ensemble candidates in STEP 04;
- XGBoost, LightGBM and CatBoost added to the tabular benchmark;
- PyTorch MLP retained as an independent neural baseline;
- training-only stratified CV with Macro-F1 mean/std;
- accuracy, balanced accuracy, macro precision/recall/F1, weighted F1,
  MCC, Top-2 accuracy, log loss, multiclass ROC-AUC;
- fit-time and inference-latency comparison;
- multi-model final test leaderboard;
- permutation feature importance;
- pre-executed notebooks with embedded plots/tables;
- dedicated synthetic-preview provenance separation;
- preview binary models intentionally excluded to prevent accidental reuse.

Preview benchmark size:
- classical candidates: 15
- final compared models including PyTorch: 16

Validation preview winner: SVM_RBF
Final preview test leader: PCA_LogisticRegression

All numeric preview results are surrogate-data portfolio previews, not official
UCI benchmark claims.


## v3 environment reliability fix

- `run_project.py` now automatically executes workflow steps with `.venv`.
- Added an early dependency check for `imblearn`, scikit-learn, PyTorch and core libraries.
- STEP 03 now gives an actionable message when `imbalanced-learn` is missing.
- Setup smoke testing now also validates XGBoost, LightGBM and CatBoost imports.
- Added `reports/TROUBLESHOOTING.md`.


## v4 robust setup bootstrap

- Transactional `.venv_build` creation prevents half-created environments.
- Detects incomplete `.venv` and recreates it automatically.
- Adds timeout protection around stdlib `venv` / `ensurepip`.
- Falls back to `virtualenv` if stdlib `venv` does not produce working pip.
- Adds `--venv-python` and `--venv-timeout` setup options.
- Improved Ctrl+C recovery and Windows troubleshooting documentation.


## v5 simplified user flow

The intended project interface is now explicitly only:

```bash
python 00_setup_project.py
python run_project.py
```

Setup owns environment creation, dependency installation and dataset acquisition.
The runner owns the complete STEP 01-09 execution and always uses the project
virtual environment automatically.
