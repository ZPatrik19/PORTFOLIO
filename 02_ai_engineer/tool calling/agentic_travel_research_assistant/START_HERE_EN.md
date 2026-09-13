# Start here — Agentic Travel Research Assistant

The normal interface is the browser UI. Terminal scripts remain available for reproducibility, automation and debugging.

## First launch on Windows

Extract the ZIP and double-click:

```text
SETUP_AND_START_UI.bat
```

The launcher:

1. creates `.venv` only if missing;
2. checks installed dependency versions without upgrading them;
3. installs only when something required is missing/incompatible;
4. registers the editable local package when needed;
5. runs a fast structure/runtime check;
6. checks router dataset/runtime metadata;
7. retrains only when the model is missing, stale or runtime-incompatible;
8. starts Streamlit.

For normal later use, double-click:

```text
RUN_UI.bat
```

## UI workflow

- **Chat** — choose a preset or ask a completely custom question.
- **Tool Explorer** — invoke one tool through a form.
- **Data Quality** — inspect quality gates and leakage/diversity reports.
- **Train & Evaluate** — retrain or benchmark from the browser.
- **Live Statistics** — see real locally persisted usage metrics.
- **Project Statistics** — explore an interactive Plotly dashboard for dataset scale, diversity, routing complexity, model diagnostics and methodology quality/latency, plus the saved PNG plot gallery.
- **Dataset Explorer** — inspect CSV data.

The sidebar also contains a high-contrast **Light / Dark analytics chart theme** selector. It applies consistently to Project Statistics, Data Quality and Live Statistics; all primary interactive charts use the same height.

## Recommended first custom question

```text
I am going to Vienna for 3 days. Check the weather, find a hotel under 150 EUR,
recommend restaurants and attractions, and show public transport options.
```

## Documentation

Start with:

- `07_docs/en/00_DOCUMENTATION_INDEX.md` — complete English documentation entry point
- `07_docs/hu/00_DOKUMENTACIOS_TERKEP.md` — Hungarian documentation entry point
- `07_docs/shared/visuals/` — shared architecture and flow diagrams

Hungarian equivalents are available next to them.
