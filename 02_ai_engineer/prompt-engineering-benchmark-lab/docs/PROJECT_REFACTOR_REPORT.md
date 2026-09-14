# Project Refactor Report

## 1. Initial State

The project was already feature-rich and working: P0–P16 prompt strategies, a bilingual Streamlit/Plotly UI, mock and cloud providers, benchmark history, custom prompts, A/B Playground, output validation, parameter sweeps, fine-tuning export and a large synthetic challenge set were present.

The main technical debt was architectural rather than functional. Runtime code lived under `04_pipeline/src`, tests depended on `pytest.ini` path injection, scripts/notebooks contained path-manipulation code, provider implementations were concentrated in a large client module, configuration paths were scattered, generated artifacts lived in several top-level directories, and environment bootstrap performed work that belonged to an explicit experiment run.

## 2. Identified Problems

- The runtime code was not a proper installable Python package.
- `sys.path`/pytest path manipulation reduced portability.
- Several presentation/provider modules had God-module characteristics.
- Six provider adapters and mock simulation were coupled in one client file.
- API retries/timeouts were not governed by one common policy.
- Configuration and path resolution were spread across scripts and UI code.
- Setup could launch an expensive full benchmark unexpectedly.
- Generated results/reports/logs were split across multiple top-level locations.
- Dependency metadata was not centralized in a modern package definition.
- Docker/Kubernetes deployment support and cross-platform launchers were incomplete.
- Documentation was extensive but partially described an older repository layout.

## 3. Architecture Changes

- Runtime code moved to the installable `03_src/prompt_benchmark` package.
- Added `pyproject.toml` with package metadata, dev tooling configuration and console entrypoint.
- Added central `ProjectPaths` repository-relative path management.
- Added validated Pydantic benchmark configuration loading.
- Split LLM adapters into `llm/providers/{gemini,groq,openai,openrouter,ollama,mock}.py` while preserving the public client interface.
- Added bounded exponential retry/jitter policy at the shared LLM client boundary.
- Added strict benchmark DataFrame schema validation.
- Consolidated generated artifacts under `07_outputs/`.
- Consolidated configuration and prompt resources under `configs/`.
- Replaced the raw localized-view `exec(compile(...))` launcher with `runpy.run_path` for trusted local view modules.
- Added a single non-UI full-pipeline entrypoint with smoke/pilot/standard/strong/full profiles.

## 4. Clean Code Improvements

- Removed runtime `sys.path` manipulation from scripts/tests/notebooks.
- Replaced many repeated path literals with `PATHS`/`ProjectPaths`.
- Introduced dedicated validation, logging and configuration modules.
- Preserved reusable prompt/evaluation logic outside notebooks and Streamlit views.
- Kept backward-compatible provider/client imports where existing code relied on them.
- Added explicit configuration/domain exception messages instead of silent failures.
- Moved runtime logging to a rotating log file plus console output.

A deliberate limitation remains: the two localized Streamlit view files are still large. The core computation has been extracted, but fully componentizing every presentation section would be a separate UI refactor with higher regression risk and limited benefit to benchmark correctness.

## 5. Testing

The project contains unit, integration, regression and edge-case tests for:

- config validation;
- paths;
- dataset splits and challenge-suite properties;
- benchmark schema validation;
- prompt strategies/custom prompts;
- parsing and structured output;
- metrics/pricing;
- provider factories and schemas;
- retry policy;
- cache/history behavior;
- bilingual UI launcher behavior;
- Playground token reshaping regression;
- Gemini adapter/config health helpers.

Final validation results are recorded at the end of the refactor process. Live-provider tests remain credential/network dependent and are not falsely marked as passing when unavailable.

## 6. Deployment

- Added Windows `run_project.bat` / `run_tests.bat`.
- Added Linux/macOS `run_project.sh` / `run_tests.sh`.
- Added Dockerfile and `.dockerignore` with a non-root runtime user.
- Added Kubernetes Namespace, ConfigMap, Secret example, Deployment and Service.
- Kubernetes readiness/liveness probes use Streamlit's native `/_stcore/health` endpoint instead of introducing an unnecessary second backend.
- Secrets are injected at runtime, never stored in the image or repository manifests.

## 7. Documentation

The repository now has structured documentation for overview, architecture, data pipeline, prompt engineering, evaluation, testing, deployment, troubleshooting and design decisions, plus this report and an interview-focused project story.

## 8. Validation Results

Validated in the refactor environment:

- **76 tests passed, 0 failed**.
- **77% configured core-package coverage**.
- Python compile/import/config checks passed.
- P0–P16 offline smoke benchmark passed.
- Standard 120-sample P0–P16 mock pipeline and report generation completed.
- 120-sample ablation and decoding-parameter sweep completed.
- All 10 notebooks completed execution; the first batch hit only the outer orchestration timeout after notebook 06, so notebooks 07–10 were executed separately and passed.
- Shell launcher syntax and Kubernetes YAML parsing passed.
- Secret and absolute-path scans found no embedded live credential or personal machine path.

Not validated because the required tools/network were unavailable: Docker image build, kubectl apply, Ruff, mypy, dependency download and live cloud-provider calls. See `docs/VALIDATION_RESULTS.md` for exact scope and limitations.

## 9. Known Limitations

- External cloud provider behavior cannot be validated without working credentials/network access.
- Synthetic challenge data is useful for controlled stress testing but is not a substitute for human-labelled production data.
- LLM provider nondeterminism cannot be eliminated by local seeds.
- Docker/Kubernetes runtime validation requires Docker/kubectl availability in the execution environment.
- Some Streamlit presentation modules remain large for compatibility.

## 10. Future Improvements

- Add CI with Ruff, mypy, pytest/coverage, Docker build and Kubernetes schema validation.
- Add human-reviewed adversarial/multilingual datasets.
- Add repeated stochastic trials and model-vs-model significance testing.
- Move run history to SQLite/PostgreSQL when concurrent multi-user use becomes a requirement.
- Further componentize the bilingual Streamlit view layer if UI maintenance becomes a dominant concern.
