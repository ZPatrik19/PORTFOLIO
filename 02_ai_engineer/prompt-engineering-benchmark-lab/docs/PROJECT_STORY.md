# Project Story – Interview Narrative

## Problem

Prompt engineering is often demonstrated with hand-picked examples. The goal of this project was to turn prompt design into an engineering experiment: hold the task and evaluation protocol constant, vary the prompt strategy, and measure quality, robustness and operational cost.

## Challenge

The difficult part was not calling an LLM. It was building a fair benchmark that avoids prompt-overfitting, survives malformed outputs/API failures, exposes uncertainty on small pilots, records tokens/latency/cost, supports multiple providers, and still remains usable from an interactive Playground.

## Approach

I separated the system into dataset preparation, prompt construction, provider adapters, parsing/validation, evaluation, persistence/history and presentation. Development/few-shot examples are separated from the final holdout. Small runs use stratified sampling over label and scenario rather than taking the first N rows. Every run persists its configuration and raw predictions so results can be audited later.

## Architecture

The installable `prompt_benchmark` package contains the reusable business logic. A provider factory exposes Mock, Ollama, Gemini, Groq, OpenRouter and OpenAI through a common response contract. Streamlit/Plotly sits on top of the same benchmark runner used by the CLI. Filesystem run history provides simple reproducibility without introducing a database prematurely.

## Engineering Decisions

- Macro F1 is a primary quality metric because each routing class should matter equally.
- Accuracy is accompanied by confidence intervals; a 10/10 pilot is not presented as proof of perfect accuracy.
- Structured-output reliability is measured separately from classification quality.
- Tokens, latency and cost are first-class metrics because the most accurate prompt may be a poor production trade-off.
- Branch-and-vote is implemented as multiple final-label branches plus aggregation, not by storing private chain-of-thought.
- A deterministic prompt-sensitive simulator exists for software testing, but its results are labelled as simulated and are not claimed as real model evidence.
- Provider adapters share timeout/retry behavior and isolate external-service failures from evaluation logic.

## Evaluation

The benchmark reports Accuracy, Wilson CI, Macro Precision/Recall/F1, bootstrap F1 CI, Weighted F1, balanced accuracy, MCC, Cohen's kappa, class-wise metrics, hard/scenario performance, output validity, JSON validity, token usage, latency percentiles, throughput and estimated cost. Paired prompt comparisons include fixed/regressed examples and McNemar testing.

## Results

Mock results demonstrate that the experiment infrastructure, difficult-case breakdowns and dashboards work end to end. Real portfolio claims should be generated with an actual model/provider using the same locked benchmark protocol. This distinction is intentional: the project avoids presenting synthetic simulator numbers as LLM quality evidence.

## Lessons Learned

Prompt quality cannot be judged from a few examples or from one aggregate score. Dataset difficulty, output contracts, latency and token overhead frequently change the engineering decision. Reproducibility also depends on recording the exact provider/model/settings/dataset snapshot, not just saving a prompt string.

## Productionization

The project evolved from a feature-rich benchmark demo into an installable package with centralized config/path management, structured logging, schema validation, retries, extensive tests, cross-platform launchers, Docker/Kubernetes deployment assets and explicit security rules for API credentials. The resulting repository can be cloned and operated without knowing the original development machine layout.
