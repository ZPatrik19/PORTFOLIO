# Project Story — Interview Narrative

## Problem
I wanted to demonstrate prompt engineering as measurable AI engineering rather than prompt collecting. The task had to expose quality, reliability, efficiency, and robustness trade-offs.

## Challenge
The difficult part was experimental integrity: leakage-safe splits, fair prompt comparisons, structured-output failures, external-provider instability, misleading tiny pilots, stale cache reuse, token/cost accounting, and preserving history across UI sessions.

## Approach
I separated data, prompt construction, provider adapters, benchmark execution, parsing/validation, evaluation, persistence, and presentation. The same core runner is reused by CLI, notebooks, and UI.

## Important engineering decisions
- Macro F1 is primary but never shown alone.
- Small perfect pilots show confidence intervals.
- Output validity is separate from semantic correctness.
- Prompt and decoding experiments are isolated.
- Mock results are explicitly simulation.
- Branch-and-vote stores final branch decisions rather than private chain-of-thought.
- Every completed run can be audited from raw predictions and manifest metadata.

## Evaluation
The system measures task quality, statistical uncertainty, scenario robustness, formatting reliability, tokens, latency, throughput, and cost. Paired comparisons expose both fixes and regressions.

## Productionization
The repository is an installable package with typed config, portable paths, provider isolation, retries, tests, cross-platform launchers, Docker/Kubernetes assets, structured documentation, and security rules.

## Lesson learned
A prompt that wins on F1 can still lose in production because of token overhead, latency, malformed output, provider support, or fragile behavior on adversarial inputs. The benchmark makes those trade-offs visible.
