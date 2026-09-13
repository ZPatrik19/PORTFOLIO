# Tool calling and tool design

This document explains the role of tool contracts, trust boundaries and the key engineering questions that should be answered when designing tool-calling systems.

## Tool catalogue

The system registers eight tools: `get_location_info`, `get_weather`, `convert_currency`, `search_hotels`, `search_attractions`, `search_restaurants`, `get_transport_options`, and `calculate`.

Every tool uses a Pydantic input model with `extra="forbid"`, preventing unexpected model-generated fields from silently reaching execution. `ToolRegistry` centralizes schema export, validation, execution, latency measurement and tracing.

The calculator does not use Python `eval()`; it evaluates a numeric AST whitelist. Weather and FX tools support explicit local/auto/live provider modes.

## Tool-calling system-design questions

The questions below are an engineering checklist for system design and review, not interview coaching.

This is not an interview-question sheet. It is an engineering checklist for designing a tool-calling or agentic system: capability boundaries, schemas, routing, orchestration, validation, failure semantics, observability, evaluation, latency and operational controls.

## 1. Capability and tool boundaries

### 1.1 What deserves its own tool?
A tool should represent a coherent, independently testable capability. Separate tools are usually appropriate when operations have different data sources, permissions, failure modes or latency profiles.

In this project `search_hotels`, `search_restaurants` and `search_attractions` are separate capabilities, while price/rating/top-k are parameters of each search tool.

### 1.2 When is granularity too fine?
If one user task requires a long chain of microscopic calls, latency and failure surface grow unnecessarily. Avoid turning every database field into a separate tool.

### 1.3 When is a tool too broad?
A single “plan_everything” tool hides failure attribution and reduces composability. Weather, accommodation and transport should not be inseparable unless there is a strong domain reason.

## 2. Schema and argument design

- Which arguments are truly required?
- What numeric ranges and type constraints are valid?
- Should a field be an enum or free text?
- Does `None` mean “no filter” or “unknown”?
- How are schema changes versioned across tests, prompts and benchmark ground truth?

Model-generated JSON is untrusted input. Function schema guidance does not replace application-side validation.

## 3. Tool selection and routing

### 3.1 LLM, rules, or a dedicated classifier?
This repository makes the trade-off measurable through a rule-based baseline, deterministic plan-then-execute routing, a supervised ML router, and optional OpenAI direct tool calling.

### 3.2 When should deterministic guardrails override a classifier?
High-precision signals such as “do not search for a hotel” are suitable for deterministic negative overrides when classifier false positives would be costly.

### 3.3 What happens under uncertainty?
Options include no call, clarification, top-label selection, thresholding, or an LLM fallback. The correct policy depends on the business cost of false positives versus false negatives.

### 3.4 Which routing metrics matter?
Exact tool-set accuracy, precision, recall, F1 and unnecessary tool-call rate provide different information. Accuracy alone is insufficient.

## 4. Argument extraction

- Is extraction performed by the same LLM, a parser, regex/heuristics, or a separate model?
- Which expected fields are actually inferable from the user text?
- How are canonical entities resolved from surface forms?
- How are Hungarian inflections such as `Bécs`, `Bécsbe`, `Bécsben`, `Bécsből` normalized?

The project's ML router deliberately separates capability classification from deterministic structured argument extraction so both can be evaluated independently.

## 5. Orchestration and dependencies

- Which calls are independent and can run in parallel?
- Which calls consume previous tool outputs?
- Is a simple ordered `PlanStep` list enough, or is a DAG/state machine required?
- What is the maximum number of agent steps?
- When should the system use plan-then-execute versus an iterative observe/act loop?

A bounded `max_steps` is essential to prevent accidental infinite loops.

## 6. Validation and trust boundaries

- Where does schema validation happen?
- What happens when arguments are invalid?
- Does a validation failure become a structured tool error or crash the run?
- Can partial invalid input reach an external provider?

This project validates tool arguments through Pydantic at the registry boundary.

## 7. Errors, timeout and retry

- Which failures are transient and retryable?
- Which failures are permanent validation failures?
- How many retries are allowed?
- Is exponential backoff required?
- What is the fallback policy?
- Is degraded/fallback mode visible in the result metadata?

The `local / auto / live` provider modes demonstrate explicit fallback semantics.

## 8. Idempotency and side effects

Current project tools are primarily read-only. For booking, payment, email or other write actions, additional questions become mandatory:

- What is the idempotency key?
- Can a retry duplicate the side effect?
- Is explicit user approval required?
- How is transaction state reconciled after a timeout?

## 9. Permissions and security

- Which users/agents may invoke which tools?
- Are secrets taken from a secret store rather than model arguments?
- Can external tool output contain prompt injection?
- What user data is allowed in traces?
- What is the retention policy for prompt/answer history?

External tool output should be treated as untrusted data, not as a new instruction layer.

## 10. Observability

Every call should be attributable through tool name, step, arguments, success/error, latency and run/correlation identity. Aggregate metrics should include tool success, p50/p95 latency, tool frequency, multi-tool rate, no-tool rate, error breakdown and common sequences.

Trace and aggregate analytics serve different purposes: traces explain individual failures; dashboards expose systemic patterns.

## 11. Evaluation

- What is the ground truth: expected tools, expected inferable arguments, task success, or all three?
- Does the corpus include single-tool, multi-tool, negation, indirect wording, noisy text and bilingual examples?
- Is there train/test template leakage?
- Is a separate challenge set used for indirect/noisy generalization?
- Are false positive calls explicitly penalized?

This project uses normalized pattern-overlap quality gates to detect leakage.

## 12. Latency and cost

- How much additional latency is acceptable for improved routing quality?
- Which tool results can be cached and with what TTL?
- Can independent calls be parallelized?
- Can unnecessary calls be removed through better routing?
- What is the p95 rather than just the mean latency?

## 13. Data freshness

- Which sources are live and which are deterministic fixtures?
- Is generation/provider timestamp recorded?
- Is synthetic inventory clearly labelled as non-bookable?
- What happens when a provider is stale or unavailable?

## 14. Testing strategy

Unit tests should cover parsers, schemas, tool logic, routing, metrics and persistence. Integration tests should cover agent → registry → tool execution, multi-tool workflows, fallbacks and persisted model compatibility. Every meaningful production bug should become a regression test.

Examples already represented in this project include Hungarian city inflection, Hungarian budget phrasing, negative hotel intent and scikit-learn model-runtime incompatibility.

## 15. Production extension questions

- Authentication and per-user history?
- Async execution and queues?
- Distributed tracing?
- Human approval before side effects?
- SLOs for individual tools?
- Retention/redaction policy?
- Provider outage strategy?
- Model/tool schema version attached to every trace?
- Rollback strategy for router models?

The key design question is not whether an LLM can call a function. It is whether capability contracts, validation, dependencies, failure semantics, evaluation and observability form a controlled system around that model decision.

## Conclusion

In a robust agent system a tool is more than a Python function: it is an explicit capability boundary, a validated contract, an observable execution unit and a security boundary.
