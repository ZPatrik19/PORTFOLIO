# Projektstruktúra

```text
prompt-engineering-benchmark-lab-refactored/
├── 00_setup/          # bootstrap / dependency helpers
├── 01_data/           # raw/mock/external/processed/fine-tuning data
├── 02_notebooks/      # 10 executable analysis notebooks
├── 03_src/            # installable prompt_benchmark package
├── 04_tests/          # pytest unit/integration/smoke suite
├── 05_scripts/        # CLI-oriented workflow scripts
├── 06_deployment/     # Docker/Kubernetes assets
├── 07_outputs/        # results/reports/logs
├── configs/           # benchmark/provider/pricing/prompt configuration
├── docs/              # bilingual documentation
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── run_project.bat / run_project.sh
└── run_tests.bat / run_tests.sh
```

## Olvasási filozófia
A számozott mappák a workflow sorrendjét teszik azonnal érthetővé, miközben a telepíthető package a kódarchitektúrát függetleníti a prezentációs sorrendtől. A generált artifactok a `07_outputs` alá kerülnek; secretek és virtual environment nem kerül Gitbe.
