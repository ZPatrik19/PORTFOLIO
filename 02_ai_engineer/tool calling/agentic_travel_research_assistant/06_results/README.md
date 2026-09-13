# Generated results

Durable, reproducible outputs retained in this repository:

- `data_quality/` — quality-gate JSON/CSV reports and diversity figures.
- `models/` — trained intent-router artifact, runtime metadata, classification report and generalization figure.
- `rule_based/` — 500-case baseline benchmark outputs.
- `plan_execute/` — 500-case deterministic planner benchmark outputs and diagnostic figures.
- `ml_router/` — 500-case ML-router benchmark outputs.
- `preset_validation/` — 30 scenarios × 2 languages × 2 offline methodologies regression results.
- `methodology_comparison.*` — side-by-side methodology comparison.
- `project_statistics/` — reproducible dataset/repository/model/benchmark statistics.
- `usage/` — documentation for local SQLite usage history; the actual `.sqlite3` file is Git-ignored.

Temporary setup/UI/notebook smoke evaluations use temporary directories and are intentionally not persisted here.

All offline benchmark runs use `TRAVEL_DATA_MODE=local` for reproducibility.
