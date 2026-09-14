# Final Project Tree

```text
prompt-engineering-benchmark-lab/
├── 00_setup/
│   ├── 00_dependency_manager.py
│   ├── 01_setup_windows.bat
│   ├── 02_run_ui.bat
│   ├── 03_run_mock_demo.bat
│   ├── 04_run_ollama_demo.bat
│   └── 05_update_environment.bat
├── 01_data/
│   ├── external/
│   ├── fine_tuning/
│   ├── mock/
│   ├── processed/
│   └── sample/
├── 02_notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_prompt_design.ipynb
│   ├── 03_benchmark.ipynb
│   ├── 04_error_analysis.ipynb
│   ├── 05_prompt_ablation.ipynb
│   ├── 06_final_analysis.ipynb
│   ├── 07_advanced_prompt_engineering.ipynb
│   ├── 08_decoding_parameters.ipynb
│   ├── 09_prompt_template_and_output_validation.ipynb
│   └── 10_finetuning_readiness.ipynb
├── 03_src/
│   └── prompt_benchmark/
│       ├── benchmark/
│       │   ├── __init__.py
│       │   └── runner.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── benchmark_suites.py
│       │   ├── mock_generator.py
│       │   ├── prepare_benchmark.py
│       │   └── validation.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── bootstrap.py
│       │   ├── diagrams.py
│       │   ├── metrics.py
│       │   ├── parsing.py
│       │   └── plots.py
│       ├── llm/
│       │   ├── providers/
│       │   │   ├── __init__.py
│       │   │   ├── gemini.py
│       │   │   ├── groq.py
│       │   │   ├── mock.py
│       │   │   ├── ollama.py
│       │   │   ├── openai.py
│       │   │   └── openrouter.py
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── client.py
│       │   ├── factory.py
│       │   ├── gemini_health.py
│       │   └── schemas.py
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── custom.py
│       │   └── strategies.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── benchmark_panel.py
│       │   ├── dataset_panel.py
│       │   ├── playground_data.py
│       │   ├── playground_panel.py
│       │   ├── playground_store.py
│       │   ├── run_history.py
│       │   └── workflow_panel.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── pricing.py
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       └── paths.py
├── 04_tests/
│   ├── conftest.py
│   ├── test_advanced_strategies.py
│   ├── test_benchmark_suites.py
│   ├── test_config_validation.py
│   ├── test_custom_prompts.py
│   ├── test_data_split.py
│   ├── test_data_validation.py
│   ├── test_expanded_challenge_set.py
│   ├── test_gemini_client_adapter.py
│   ├── test_gemini_env_config.py
│   ├── test_gemini_health.py
│   ├── test_metrics.py
│   ├── test_mock_data.py
│   ├── test_mock_prompt_sensitivity.py
│   ├── test_openai_schema.py
│   ├── test_parsing.py
│   ├── test_paths.py
│   ├── test_playground_token_reshape.py
│   ├── test_pricing.py
│   ├── test_prompts.py
│   ├── test_provider_adapters_unit.py
│   ├── test_provider_factory.py
│   ├── test_provider_schemas.py
│   ├── test_retry_policy.py
│   ├── test_runner_cache.py
│   ├── test_ui_language_switch.py
│   └── test_workflow_history.py
├── 05_scripts/
│   ├── 00_run_pipeline.py
│   ├── 01_generate_mock_data.py
│   ├── 02_prepare_data.py
│   ├── 03_check_provider.py
│   ├── 04_run_benchmark.py
│   ├── 05_run_ablation.py
│   ├── 06_run_parameter_sweep.py
│   ├── 07_generate_report.py
│   ├── 08_generate_diagrams.py
│   ├── 09_run_smoke_test.py
│   ├── 10_run_free_demo.py
│   ├── 11_ui_app.py
│   ├── 11_ui_view_en.py
│   ├── 11_ui_view_hu.py
│   ├── 12_export_finetuning_data.py
│   ├── 13_api_integration_example.py
│   ├── 14_run_custom_prompt.py
│   └── 15_test_gemini_connection.py
├── 06_deployment/
│   ├── docker-entrypoint.sh
│   └── kubernetes/
│       ├── README.md
│       ├── configmap.yaml
│       ├── deployment.yaml
│       ├── namespace.yaml
│       ├── secret.example.yaml
│       └── service.yaml
├── 07_outputs/
│   ├── logs/
│   ├── reports/
│   │   └── figures/
│   └── results/
│       ├── history/
│       ├── parameter_sweeps/
│       └── raw/
├── configs/
│   ├── prompts/
│   │   ├── custom/
│   │   ├── examples/
│   │   ├── hu/
│   │   ├── playground/
│   │   ├── templates/
│   │   └── p0...p16 prompt templates
│   ├── benchmark.yaml
│   ├── parameter_sweeps.yaml
│   ├── pricing.yaml
│   └── providers.yaml
├── docs/
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_DATA_PIPELINE.md
│   ├── 04_PROMPT_ENGINEERING.md
│   ├── 05_EVALUATION.md
│   ├── 06_TESTING.md
│   ├── 07_DEPLOYMENT.md
│   ├── 08_TROUBLESHOOTING.md
│   ├── 09_DESIGN_DECISIONS.md
│   ├── PROJECT_REFACTOR_REPORT.md
│   ├── PROJECT_STORY.md
│   ├── VALIDATION_RESULTS.md
│   ├── FINAL_PROJECT_TREE.md
│   └── additional HU/EN guides
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── README_HU.md
├── RUN_UI.bat
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── run_project.bat
├── run_project.sh
├── run_tests.bat
└── run_tests.sh
```

Generated CSV/PNG/history artifacts inside `01_data/` and `07_outputs/` are intentionally summarized rather than listing thousands of experiment files individually.
