# Mock Simulation & Metrics

## Purpose
The mock provider is a deterministic prompt-sensitive simulator that allows the complete benchmark/UI/reporting stack to be exercised without a key or model runtime.

## What it simulates
Prompt features influence deterministic success probability and simulated token/latency overhead. Advanced prompts can therefore look stronger but more expensive. This is intentional software-demo behavior, not model evidence.

## Explicit telemetry labels
Mock rows identify `token_source = estimated_mock` and `latency_source = simulated_mock` so simulated values cannot be confused with provider-reported usage.

## What becomes real with a provider
Predictions, provider-reported usage where available, wall-clock request latency, SDK/API errors, and actual parser validity come from the real run.

## Why keep the simulator
It gives deterministic regression coverage for dashboards, histories, metrics, error analysis, cache, and prompt-strategy workflows without quota/cost/network dependencies.
