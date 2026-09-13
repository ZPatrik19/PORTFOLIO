# Architecture and data flow

This chapter describes the technical structure of the system and the complete lifecycle of a request.

## 1. System boundary

The input is a natural-language travel request. The output is a human-readable answer backed by structured tool calls, validation and a trace.

## 2. Runtime flow

```text
User / Streamlit UI
        ↓
AgentService
        ↓
Routing / Planner
        ↓
ToolRegistry
        ↓
Pydantic validation
        ↓
Tool execution
        ↓
Local CSV / Live API
        ↓
Structured result + trace
        ↓
Answer + usage analytics
```

All routing methodologies use the same `ToolRegistry`. Therefore rule-based, plan-execute, ML-router and OpenAI-direct paths execute the same tool contracts.

## 3. Major components

- `08_ui/app.py`: browser UI.
- `agent/service.py`: unified agent entry point.
- `agent/*`: routing and orchestration strategies.
- `tools/*`: tool contracts and execution.
- `data_store.py`: local data access and cached city lookup.
- `training/*`: router training, runtime metadata and freshness checks.
- `evaluation/*`: benchmarks and metrics.
- `usage/*`: SQLite-backed live usage analytics.

## 4. Provider modes

`local` uses deterministic local data only. `auto` tries live providers and falls back locally. `live` requires the live provider to succeed. Weather can use Open-Meteo and currency conversion can use Frankfurter. Hotel, restaurant and attraction inventories are synthetic and must not be interpreted as live availability.

## 5. Dependencies and parallelism

Independent calls can in principle run in parallel. The calculator can also become a downstream dependency when one tool result feeds a later arithmetic step. The current implementation favors explicit, inspectable orchestration over premature asynchronous complexity.

## 6. Observability

Every tool call records its tool name, arguments, output, latency and success/error state. Completed Chat runs persist summary metadata in SQLite, which powers the Live Statistics page.

## 7. Diagrams

- `../shared/visuals/01_runtime.png` — runtime components.
- `../shared/visuals/02_data_flow.png` — data flow.
- `../shared/visuals/03_agent_loop.png` — agent loop.
- `../shared/visuals/04_provider_modes.png` — provider modes.
- `../shared/visuals/06_ui_flow.png` — UI flow.

## Conclusion

The architecture is intentionally modular: routing, tool contracts, data providers, evaluation and UI are separate layers. This allows multiple agent strategies to be evaluated against the same execution layer.
