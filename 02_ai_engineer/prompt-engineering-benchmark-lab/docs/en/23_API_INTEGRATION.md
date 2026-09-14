# API Integration

## Unified boundary
All providers implement the same client contract and return normalized telemetry. The benchmark does not know provider-specific SDK response shapes.

## Request lifecycle
```text
PromptPayload -> provider adapter -> timeout/retry -> provider response -> normalized LLMResponse -> parser/validation -> benchmark row
```

## Error classes
Authentication/configuration errors fail fast. Timeouts, selected rate-limit responses, and transient server failures may be retried with bounded backoff. Malformed model output is recorded as a benchmark result/error state rather than hidden.

## Token and latency sources
Real providers use reported token usage where available and wall-clock request latency. Mock mode labels estimated/simulated sources explicitly.

## Connection testing
The UI/provider-check scripts allow a small request before launching a benchmark. This is the recommended place to detect model-name, credential, quota, and endpoint problems.
