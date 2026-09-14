# UI Workflow

## Recommended order
1. **Provider / API** — choose Mock/Ollama/cloud provider, model, and runtime key if required; test connection.
2. **Dataset** — inspect source, size, class balance, scenario distribution, and difficulty profile.
3. **Playground** — develop prompts with classification, free-text generation, or A/B comparison; save presets.
4. **Benchmark** — select run profile and strategies; monitor live progress, tokens, latency, and partial leaderboard.
5. **Output Validation** — inspect confusion matrix, invalid outputs, JSON/contract validity, and errors for the selected run.
6. **Dashboard** — compare quality, uncertainty, tokens, latency, cost, difficulty, and scenario heatmaps.
7. **History / Comparison** — reload prior runs and compare model/provider/prompt experiments.

## State persistence
Runtime UI state is kept in Streamlit session state; completed benchmark runs are also persisted to disk with a run ID, manifest, dataset snapshot, raw predictions, and summary metrics.

## Playground A/B
Classification A/B compares two real prompt strategies on the same ticket and optionally a known ground truth. Generative A/B compares editable templates and records output length, tokens, latency, keyword/format adherence, and JSON validity where relevant.
