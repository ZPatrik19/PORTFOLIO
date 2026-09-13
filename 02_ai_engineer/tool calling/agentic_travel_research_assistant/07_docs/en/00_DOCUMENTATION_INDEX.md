# Documentation index

This page is the entry point for the English technical documentation.

The documentation is organized as an end-to-end engineering narrative. For a first read, follow the numbered files in order.

| # | Document | Purpose |
|---|---|---|
| 01 | [Project overview](01_PROJECT_OVERVIEW.md) | objective, use case, main components |
| 02 | [Architecture and data flow](02_ARCHITECTURE_AND_DATA_FLOW.md) | runtime components, request flow, provider modes |
| 03 | [Data and data quality](03_DATA_AND_DATA_QUALITY.md) | datasets, generation, leakage, quality gates |
| 04 | [Tool calling and tool design](04_TOOL_CALLING_AND_TOOL_DESIGN.md) | contracts, schemas, trust boundaries, design questions |
| 05 | [Agent routing and orchestration](05_AGENT_ROUTING_AND_ORCHESTRATION.md) | rule-based, plan-execute, ML router, OpenAI direct |
| 06 | [Model training and compatibility](06_MODEL_TRAINING_AND_COMPATIBILITY.md) | TF-IDF router, splits, thresholds, persistence |
| 07 | [Evaluation and statistics](07_EVALUATION_AND_STATISTICS.md) | metrics, benchmarks, live/project statistics |
| 08 | [UI and usage](08_UI_AND_USAGE.md) | setup, Streamlit tabs, custom questions, presets |
| 09 | [Code reference and repository audit](09_CODE_REFERENCE_AND_REPOSITORY_AUDIT.md) | file responsibilities and repository decisions |
| 10 | [Limitations and next steps](10_LIMITATIONS_AND_NEXT_STEPS.md) | current scope and production extensions |
| 11 | [Preset questions](11_PRESET_QUESTIONS.md) | 30 bilingual scenarios and regression role |
| 12 | [Visualization system](12_VISUALIZATION_SYSTEM.md) | Light/Dark themes, contrast, equal chart height and the shared Plotly rendering pipeline |

Shared diagrams live under `../shared/visuals/`.

## Recommended reading order

Read 01 through 12 for a complete walkthrough. For maintenance work, jump directly to the topic-specific document.
