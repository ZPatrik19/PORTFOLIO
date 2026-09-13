# Agent routing and orchestration

This document explains how the system decides which tools are required and how routing decisions become executable workflows.

## 1. Why multiple routing strategies?

The project is designed for comparison. Multiple decision strategies use the same tool layer, allowing routing quality to be separated from execution quality.

## 2. `rule_based`

A lexical/regex baseline. It is fast and deterministic but generalizes poorly. Its main role is to provide a reference point.

## 3. `plan_execute`

The system first creates an explicit plan and then executes steps. Much of the Hungarian morphology and argument parsing lives in `heuristics.py`. The approach is inspectable, but manual rules have maintenance cost.

## 4. `ml_router`

A multi-label classifier selects tool intents while arguments come from deterministic parsing. High-precision guardrails can override classifier false positives, especially around explicit negation.

## 5. `openai_direct`

The LLM receives tool schemas and returns function-call objects. Python validates and executes each call, returns the tool output to the model, and repeats until the model produces a final answer or the max-step limit is reached.

## 6. Composite workflows

Not every capability should be represented as a classifier label. A budget workflow can be deterministic: hotel/food/transport results → `calculate`. This keeps dependencies explicit and avoids unnecessary label-space complexity.

## 7. Negation and guardrails

Requests such as “do not search hotels” or “accommodation is already booked” are negative routing signals. Explicit guardrails are useful because classifier probability alone is not a sufficient contract for preventing unnecessary or side-effecting calls.

## Conclusion

All routing strategies operate over the same tool layer. That makes methodology benchmarks meaningful: observed differences originate in decision logic rather than different tool implementations.
