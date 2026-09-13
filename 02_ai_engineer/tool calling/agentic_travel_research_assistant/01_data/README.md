# Data layer

This repository contains deterministic synthetic travel inventories plus labelled bilingual language corpora for routing and end-to-end agent evaluation.

The full reproducible regeneration path is:

```powershell
python 00_setup/04_generate_data.py
python 00_setup/05_upgrade_data_quality.py
```

The second step is important: it replaces the intentionally simple base templates with the diversified corpus used by the final project.

## raw/

- `cities.csv` — 60 destination profiles.
- `hotels.csv` — 180,000 synthetic hotel records.
- `attractions.csv` — 90,000 synthetic POI records.
- `restaurants.csv` — 90,000 synthetic restaurant records.
- `transport.csv` — 60 transport profiles.
- `weather_fallback.csv` — 7,200 deterministic weather rows.
- `fx_rates_fallback.csv` — 30-currency fallback snapshot.
- `sample_user_queries.csv` — 90,000 diverse HU/EN requests.
- `intent_router_dataset.csv` — 240,000 labelled multi-label examples: 192k train / 24k validation / 24k test.
- `intent_router_challenge.csv` — 36,000 indirect/noisy/negated examples.

## benchmark/

- `agent_tasks.json` — 22,500 held-out-surface end-to-end tasks with expected tool calls and arguments.
- `agent_tasks.jsonl` — the same benchmark as JSON Lines.

## processed/

- `dataset_manifest.json` — row counts, split sizes and provenance notes.

## Quality policy

The inventory tables are synthetic by design and are not real/bookable businesses. Their purpose is deterministic retrieval/filtering/ranking testing.

The language corpus uses split-specific surface phrase families. After city/number/currency slots are normalized, train/test overlap is audited and must remain at zero under the current quality gate. Exact duplicate queries are also rejected by the generator.

Run the audit from the browser UI or through `travel_agent.quality.audit_all`.
