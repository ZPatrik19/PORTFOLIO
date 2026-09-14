# Security & Reproducibility

## Secret handling
- `.env` is ignored by Git.
- `.env.example` contains placeholders only.
- provider keys are never embedded in prompt templates, result CSV files, Docker images, or Kubernetes example YAML.
- authentication errors are not retried.

## Reproducibility controls
- fixed dataset/split seed;
- deterministic representative pilot sampling;
- development/holdout/few-shot separation;
- run manifests with provider/model/settings/dataset identity;
- raw prediction checkpoints;
- cache invalidation when data or sampling settings change;
- cost recomputation when pricing changes without re-calling the provider.

## Nondeterministic boundary
A real LLM provider can remain nondeterministic despite local seeds. Reproducibility therefore means preserving exact inputs, prompt version, provider/model identifier, sampling parameters, response telemetry, and dataset snapshot—not promising bitwise-identical cloud responses.
