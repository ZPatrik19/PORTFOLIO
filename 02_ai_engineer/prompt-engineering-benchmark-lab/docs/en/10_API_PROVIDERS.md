# API Providers

## Supported providers
- **Mock:** deterministic offline prompt-sensitive simulator; no key, no real model quality claim.
- **Ollama:** local model runtime; no cloud key, hardware-dependent latency.
- **Gemini:** cloud provider through Google GenAI SDK.
- **Groq:** cloud inference provider with its own capability/rate-limit profile.
- **OpenRouter:** multi-model routing provider.
- **OpenAI:** cloud provider with structured-output support depending on model/API capabilities.

## Common adapter contract
Every provider is normalized into the same response object: raw output, parsed label, input/output tokens, latency, validity flags, error information, model/provider metadata, and optional cost.

## Reliability policy
- timeout is explicit;
- transient provider failures can use bounded exponential backoff;
- authentication/invalid-request errors are fail-fast;
- malformed responses are surfaced as structured errors, not swallowed;
- benchmark checkpoints allow resume after quota/rate-limit interruption.

## Secrets
Keys are runtime secrets. `.env` is ignored; `.env.example` contains placeholders only. The UI may accept a key for the current process, but it must not be written to source, result CSV, logs, Dockerfile, or Kubernetes example manifests.
