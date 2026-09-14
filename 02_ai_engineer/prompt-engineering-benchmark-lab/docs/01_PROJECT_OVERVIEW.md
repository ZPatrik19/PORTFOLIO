# Project Overview

## Problem
Prompt engineering is often evaluated by visually inspecting a few responses. That is not sufficient for an engineering decision: prompt changes can improve task quality while increasing output failures, token usage, latency, or provider cost.

This project turns prompt engineering into a controlled experiment on support-ticket intent classification. The same labelled inputs are executed through multiple prompt architectures and provider adapters, then evaluated with reproducible metrics and error analysis.

## Objective
Measure which prompt strategy provides the strongest **quality / reliability / latency / token / cost** trade-off under controlled conditions.

The primary benchmark compares P0–P16 prompt strategies. Additional experiments cover decoding parameters, custom prompts, structured output, branch-and-vote execution, output validation, fine-tuning data readiness, and free-form generation in the Playground.

## Engineering objective
The portfolio value is not the classifier itself. It is the complete AI engineering workflow:

1. deterministic dataset preparation and leakage-safe splits;
2. versioned prompt strategies and custom prompt presets;
3. provider-neutral LLM adapters;
4. resumable request-level benchmark execution;
5. structured-output validation;
6. token/latency/cost observability;
7. statistical and per-scenario evaluation;
8. interactive Streamlit/Plotly analysis;
9. benchmark history and reproducible manifests;
10. tests, packaging, Docker and Kubernetes deployment.

## Approach

```text
Dataset source
    ↓
Schema validation + deterministic split
    ↓
Prompt strategy / custom prompt
    ↓
Provider-neutral LLM client
    ↓
Request execution + retry + checkpoint
    ↓
Output parsing / contract validation
    ↓
Quality + reliability + efficiency metrics
    ↓
Dashboard / reports / history
```

The built-in mock provider is a deterministic prompt-sensitivity simulator. It exists to validate the software pipeline offline and must not be presented as real LLM evidence. Real portfolio claims should come from Ollama or a configured cloud provider.

## Technologies

| Technology | Why it is used |
|---|---|
| Python | Core implementation and reproducible experimentation |
| pandas / NumPy | Dataset and result processing |
| scikit-learn | Classification metrics and reports |
| Pydantic | Critical configuration validation |
| Streamlit | Interactive experiment UI |
| Plotly | Interactive benchmark/dashboard visualizations |
| Matplotlib | Durable report/README figures |
| Hugging Face Datasets | Optional independent benchmark source |
| OpenAI / Groq / Gemini / OpenRouter / Ollama adapters | Provider comparison behind one interface |
| pytest / coverage | Regression and integration validation |
| Ruff / mypy configuration | Static quality tooling |
| Docker / Kubernetes | Portable execution/deployment |

## Evaluation
The system records both task quality and engineering cost:

- accuracy and Wilson 95% confidence interval;
- macro precision / recall / F1;
- weighted F1, balanced accuracy, MCC, Cohen's kappa;
- per-class metrics and confusion matrices;
- invalid output / JSON / output-contract validity;
- bootstrap confidence intervals;
- easy/medium/hard and scenario-level metrics;
- fixed vs regressed samples against P0;
- token usage and tokens per correct prediction;
- P50/P95/P99 latency and throughput;
- estimated cost when configured.

## Results
Generated results live under `07_outputs/results/` and selected durable reports under `07_outputs/reports/`. Mock demo values are explicitly labelled simulation. Real-provider results are intentionally not hard-coded into project documentation because model versions, quotas and prices can change.

## Limitations
- The bundled Challenge Set is synthetic, although intentionally adversarial and diverse.
- External provider behavior and model availability change over time.
- API-side nondeterminism can remain even with deterministic client settings.
- Cost metrics are estimates based on `configs/pricing.yaml`; free tiers and real invoices can differ.
- The Streamlit presentation layer is intentionally UI-oriented and therefore less amenable to unit coverage than the benchmark core.

## Possible improvements
- Human-labelled adversarial holdout set.
- Multiple independent real datasets and languages.
- Provider contract tests in CI with opt-in credentials.
- Persistent object storage/database for multi-user benchmark history.
- Distributed/concurrent benchmark executor with provider-aware rate limiting.
- Model-specific tokenizers for pre-run cost estimation.
