# Complete Test Catalog

This document describes every pytest module and test case in the repository. The goal is not merely to show that tests exist, but to make explicit which failure mode or regression each check protects against.


## Test layers

- `unit`: fast isolated logic; no real external service.
- `integration`: multiple components together; no paid/live cloud call.
- `smoke`: minimal end-to-end offline pipeline.
- `external`: reserved marker for opt-in credential/network tests; none are required by the default suite.

## Commands

```bash
pytest
pytest -m unit
pytest -m integration
pytest -m smoke
pytest --cov=prompt_benchmark --cov-report=term-missing
```

## `test_advanced_strategies.py`

**Layer:** `unit`  
**Purpose:** Advanced prompt-strategy contracts, branch-and-vote behavior, structured output, prompt isolation, and provider capability checks.

- `test_advanced_strategy_count_and_names` — Verifies that the advanced strategy registry exposes the expected number and stable strategy identifiers.
- `test_tree_strategy_has_three_independent_branches` — Verifies that the Tree-of-Thought-inspired branch-and-vote strategy executes three independent decision branches rather than a single hidden reasoning trace.
- `test_full_template_uses_structured_output` — Verifies that the full advanced template requests structured output so downstream parsing has a stable contract.
- `test_mock_extracts_delimited_ticket_not_prompt_definitions` — Prevents the mock classifier from accidentally classifying label definitions or instructions instead of the delimited user ticket.
- `test_provider_capability_matrix_exposes_top_k_support` — Checks that provider capability metadata clearly states whether top_k is supported.
- `test_unsupported_groq_top_k_fails_before_api_call` — Ensures unsupported Groq top_k configuration fails locally before any external request is sent.
- `test_prompt_template_file_exists` — Ensures the reusable prompt-template resource required by the UI and documentation is packaged in the repository.

## `test_benchmark_suites.py`

**Layer:** `unit`  
**Purpose:** Benchmark-suite normalization, profiling, balanced/representative sampling, identifier preservation, and duplicate-rate reporting.

- `test_normalize_custom_csv_schema` — Checks that uploaded/custom CSV data is normalized into the canonical benchmark schema.
- `test_profile_reports_six_classes` — Ensures dataset profiling reports all six supported intent classes.
- `test_balanced_sample_is_balanced` — Verifies that balanced sampling preserves equal class representation.
- `test_representative_subset_covers_all_labels_and_multiple_scenarios` — Checks that pilot subsets cover every label and multiple scenario families instead of taking only easy leading rows.
- `test_normalize_preserves_unique_sample_ids` — Ensures valid user-provided sample identifiers survive normalization unchanged.
- `test_profile_reports_raw_duplicate_rate_before_cleaning` — Ensures duplicate-rate telemetry is calculated on raw input before deduplication can hide the problem.

## `test_cli.py`

**Layer:** `integration`  
**Purpose:** Installed CLI argument forwarding and user-facing command validation.

- `test_cli_forwards_workflow_flags` — Verifies that CLI flags such as provider, strategy, and limit are forwarded unchanged to workflow scripts.
- `test_cli_rejects_unknown_command` — Ensures unknown CLI commands fail with a non-zero exit code rather than being silently ignored.

## `test_config_validation.py`

**Layer:** `unit`  
**Purpose:** Configuration loading, Pydantic validation, and actionable failure messages.

- `test_benchmark_config_loads_and_validates` — Loads the benchmark YAML through the typed config model and validates required values.
- `test_missing_config_has_actionable_error` — Ensures a missing config file produces a clear, actionable exception.
- `test_invalid_yaml_has_actionable_error` — Ensures malformed YAML produces a clear configuration error rather than an opaque stack trace.

## `test_custom_prompts.py`

**Layer:** `integration`  
**Purpose:** Custom prompt rendering, validation, persistence, and prompt-sensitive mock behavior.

- `test_custom_prompt_renders_ticket` — Checks that a custom prompt correctly interpolates the ticket payload.
- `test_custom_prompt_requires_ticket_placeholder` — Rejects custom templates that cannot inject the benchmark ticket, preventing meaningless runs.
- `test_custom_prompt_save_and_load` — Verifies lossless persistence and reload of custom prompt presets.
- `test_mock_custom_prompt_detects_advanced_components` — Ensures the prompt-sensitive mock simulator recognizes advanced components in user-defined prompts.

## `test_data_split.py`

**Layer:** `unit`  
**Purpose:** Leakage-safe, balanced, disjoint development/holdout/few-shot data splits.

- `test_splits_are_disjoint_and_balanced` — Ensures development, holdout, and few-shot sets are balanced and contain no overlapping samples.

## `test_data_validation.py`

**Layer:** `unit`  
**Purpose:** Strict benchmark schema validation before any paid/external request can be sent.

- `test_valid_benchmark_schema_passes` — Confirms a correctly shaped benchmark frame is accepted.
- `test_duplicate_ids_fail_before_api_calls` — Rejects duplicate sample IDs before any provider request can consume quota or money.
- `test_unknown_label_fails_before_api_calls` — Rejects labels outside the supported intent vocabulary before inference starts.

## `test_expanded_challenge_set.py`

**Layer:** `integration`  
**Purpose:** Scale and scenario diversity of the expanded adversarial Challenge Set.

- `test_expanded_challenge_has_eighteen_scenario_families` — Ensures the expanded challenge benchmark contains all 18 designed robustness scenario families.
- `test_default_generator_is_10x_scale` — Ensures the default synthetic generator produces the intended 10x-scale dataset.

## `test_gemini_client_adapter.py`

**Layer:** `unit`  
**Purpose:** Gemini adapter request construction and interaction payload compatibility.

- `test_gemini_adapter_builds_interactions_request` — Checks that the Gemini adapter builds the expected interactions request payload for the SDK boundary.

## `test_gemini_env_config.py`

**Layer:** `unit`  
**Purpose:** Single-UI launcher policy, shared runtime controls, and removal of obsolete launch paths.

- `test_root_launchers_are_explicit_and_keep_single_ui_entrypoint` — Ensures root launchers delegate to one canonical Streamlit entrypoint.
- `test_unified_ui_contains_language_provider_and_api_key_controls` — Ensures language, provider, model, and API-key controls live in the shared UI runtime layer.
- `test_obsolete_language_and_gemini_config_launchers_are_removed` — Prevents obsolete HU/EN/Gemini-specific launchers from reappearing and fragmenting UX.

## `test_gemini_health.py`

**Layer:** `unit`  
**Purpose:** Gemini health-check response parsing, usage accounting, and missing-key behavior.

- `test_gemini_health_parses_response_and_usage` — Checks that Gemini health responses are parsed into model output and token-usage telemetry.
- `test_gemini_health_requires_key` — Ensures Gemini health checks fail clearly when no API key is configured.

## `test_metrics.py`

**Layer:** `unit`  
**Purpose:** Classification metrics, invalid predictions, and uncertainty for small perfect samples.

- `test_perfect_metrics` — Checks the metric implementation against a perfect six-class prediction case.
- `test_confusion_table_handles_invalid_nan_predictions` — Ensures invalid/NaN predictions are represented safely in the confusion table instead of crashing sklearn.
- `test_wilson_interval_keeps_small_perfect_sample_uncertain` — Ensures a tiny 100% pilot still shows statistical uncertainty instead of implying proven perfection.

## `test_mock_data.py`

**Layer:** `unit`  
**Purpose:** Balance and minimum scale of the offline synthetic dataset generator.

- `test_mock_generator_is_balanced_and_large_enough_for_default_splits` — Ensures the synthetic source is balanced and large enough to create the configured development/holdout/few-shot sets.

## `test_mock_prompt_sensitivity.py`

**Layer:** `unit`  
**Purpose:** Realistic case-type coverage and deterministic quality/cost trade-offs in the mock simulator.

- `test_mock_dataset_contains_all_realistic_case_types` — Checks that the prompt-sensitivity simulator sees the complete set of designed realistic case types.
- `test_advanced_prompt_is_better_but_more_expensive_in_mock_simulation` — Confirms the mock simulator encodes a deterministic quality-versus-token/latency trade-off for advanced prompts.

## `test_openai_schema.py`

**Layer:** `unit`  
**Purpose:** Strict JSON Schema contract used for provider-enforced structured output.

- `test_structured_output_schema_is_strict_and_label_constrained` — Ensures the OpenAI-style JSON Schema allows only the six known labels and rejects extra properties.

## `test_parsing.py`

**Layer:** `unit`  
**Purpose:** Plain-label and JSON response parsing into the common classification contract.

- `test_plain_label_parsing` — Checks parsing of a simple one-label model response.
- `test_json_parsing` — Checks parsing and validation of JSON-formatted classification output.

## `test_paths.py`

**Layer:** `unit`  
**Purpose:** Portable repository-root discovery and environment override behavior.

- `test_project_paths_point_to_repository_layout` — Verifies canonical ProjectPaths resolve to the expected repository directories.
- `test_project_root_env_override` — Verifies PROMPT_BENCHMARK_ROOT can explicitly select a valid runtime repository root.
- `test_path_methods_are_called_after_path_composition` — Regression guard for pathlib precedence: prevents expressions such as `root / "file.csv".exists()` where `.exists()` is accidentally invoked on a string instead of the composed `Path`.

## `test_playground_store.py`

**Layer:** `unit`  
**Purpose:** Robust Playground history loading in the presence of missing/corrupted JSONL rows.

- `test_load_playground_history_skips_malformed_rows` — Ensures one corrupted history row is logged/skipped without breaking the complete Playground history load.
- `test_load_playground_history_missing_file_returns_empty` — Ensures a first-run missing Playground history file is treated as an empty history, not an application error.

## `test_playground_token_reshape.py`

**Layer:** `unit`  
**Purpose:** Regression tests for A/B Playground token reshaping and the previous pandas.melt column collision.

- `test_reshape_token_usage_allows_existing_total_tokens_column` — Regression test for the former pandas.melt value_name="tokens" collision in classification A/B charts.
- `test_reshape_token_usage_generation_shape` — Checks the long-form token dataframe used by generative A/B charts.

## `test_pricing.py`

**Layer:** `unit`  
**Purpose:** Token-based request cost calculation.

- `test_cost_calculation` — Checks input/output token pricing arithmetic against a known expected cost.

## `test_prompts.py`

**Layer:** `unit`  
**Purpose:** Core prompt-strategy rendering and structured-output strategy behavior.

- `test_all_strategies_render_ticket` — Ensures every built-in strategy can render the ticket without losing user input.
- `test_structured_output_strategy` — Checks that the structured-output strategy advertises and produces the expected output contract.

## `test_provider_adapters_unit.py`

**Layer:** `unit`  
**Purpose:** Provider adapter success paths and credential guards without making live paid calls.

- `test_openai_adapter_success` — Exercises the OpenAI adapter success path with a mocked SDK response and verifies normalized telemetry.
- `test_openrouter_adapter_success` — Exercises the OpenRouter adapter success path without a live network call.
- `test_groq_adapter_success` — Exercises the Groq adapter success path without a live network call.
- `test_ollama_adapter_success` — Exercises the Ollama adapter success path without requiring a locally running model.
- `test_cloud_adapters_require_keys` — Ensures cloud adapters reject missing credentials before attempting network I/O.

## `test_provider_factory.py`

**Layer:** `integration`  
**Purpose:** Provider registry completeness and credential-free mock client creation.

- `test_expected_providers_are_available` — Checks that Mock, Ollama, Gemini, Groq, OpenRouter, and OpenAI are all registered.
- `test_mock_factory_requires_no_credentials` — Ensures the offline mock provider can always be created without secrets.

## `test_provider_schemas.py`

**Layer:** `unit`  
**Purpose:** Strict schema shape for Groq structured-output requests.

- `test_groq_structured_output_schema_is_strict` — Ensures Groq schema-constrained output uses a strict label enum and disallows unexpected fields.

## `test_retry_policy.py`

**Layer:** `unit`  
**Purpose:** Transient-error retry classification and fail-fast authentication behavior.

- `test_retryable_provider_errors` — Checks that transient timeout/rate-limit/server errors are classified as retryable.
- `test_authentication_errors_are_not_retried` — Ensures invalid credentials fail immediately instead of wasting retries and quota.

## `test_runner_cache.py`

**Layer:** `integration`  
**Purpose:** Benchmark-cache correctness across changed data, profiles, sampling settings, and pricing updates.

- `test_cache_rejects_same_sample_id_with_different_text` — Prevents stale cached predictions from being reused when a sample ID now maps to different content.
- `test_cache_rejects_rows_outside_changed_run_profile` — Prevents a full-run cache from contaminating a smaller/different pilot profile.
- `test_cache_rejects_sampling_setting_change_to_default` — Invalidates cached predictions when decoding settings change, including transitions back to defaults.
- `test_cached_cost_is_recomputed_when_pricing_changes` — Reuses valid predictions but recomputes monetary cost when pricing configuration changes.
- `test_cache_accepts_numeric_equivalent_sampling_settings` — Treats numerically equivalent settings such as 40 and 40.0 as the same cache identity.

## `test_smoke_pipeline.py`

**Layer:** `smoke`  
**Purpose:** Minimal end-to-end offline benchmark execution through validation, prompting, inference, and metrics.

- `test_minimal_end_to_end_benchmark_smoke` — Runs the smallest complete offline path from validated data through prompt rendering, mock inference, checkpointing, and evaluation.

## `test_ui_language_switch.py`

**Layer:** `integration`  
**Purpose:** Bilingual single-UI wiring, shared runtime controls, and feature parity between HU/EN views.

- `test_bilingual_ui_launcher_exists_and_loads_both_views` — Ensures the shared launcher can load both Hungarian and English views.
- `test_language_views_do_not_reset_page_config` — Prevents language view modules from reinitializing global Streamlit page configuration.
- `test_ui_bat_delegates_to_single_project_launcher` — Ensures the legacy RUN_UI.bat delegates to the canonical project launcher.
- `test_runtime_controls_live_only_in_shared_launcher` — Prevents provider/model/API-key controls from being duplicated in language-specific view modules.
- `test_both_languages_expose_custom_prompt_ui` — Checks HU and EN views both expose custom-prompt functionality.

## `test_workflow_history.py`

**Layer:** `integration`  
**Purpose:** Persistent benchmark-run history and Playground preset round trips.

- `test_run_history_round_trip` — Verifies benchmark run manifests/results can be saved and loaded without losing metadata.
- `test_generation_preset_round_trip` — Verifies generative Playground presets can be saved and restored losslessly.
