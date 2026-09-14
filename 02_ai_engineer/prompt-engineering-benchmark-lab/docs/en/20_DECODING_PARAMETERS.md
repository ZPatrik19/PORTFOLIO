# Decoding Parameters

Prompt content and decoding settings are separate independent variables. The parameter lab therefore holds the prompt fixed while sweeping one variable at a time.

## Temperature
Controls randomness. Classification generally favors low values for stability. Example sweep: `0.0, 0.2, 0.5, 0.8`.

## Top-p
Nucleus sampling threshold. Example sweep: `0.5, 0.8, 0.95, 1.0`.

## Top-k
Restricts candidate tokens to the K most likely options. Example sweep: `10, 20, 40, 80`. Not all providers expose it.

## Capability-aware behavior
Provider capability metadata prevents unsupported fields from being sent. A uniform UI does not imply uniform API capabilities.

## Interpretation
For deterministic classification, quality/reliability degradation at more stochastic settings is often more important than creative diversity. For free-text generation, the trade-off can be different.
