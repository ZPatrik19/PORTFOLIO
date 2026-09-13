# Environment setup

Python 3.10+ is required.

## Recommended Windows startup

Use:

```text
SETUP_AND_START_UI.bat
```

The launcher is intentionally conservative:

1. it creates `.venv` only if it does not exist;
2. `00_setup/06_check_dependencies.py` checks installed versions without modifying anything;
3. if every dependency already satisfies the project constraints, dependency installation and upgrades are skipped;
4. the local editable project registration is done only once per virtual environment;
5. `00_setup/07_prepare_if_needed.py` retrains the ML router only when the model is missing or stale relative to the training-dataset hash;
6. otherwise the existing environment and model are reused.

For normal later starts, use:

```text
RUN_UI.bat
```

This only activates the existing environment and starts Streamlit.

## Manual setup

For a new environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python 00_setup/06_check_dependencies.py
```

Only if that check reports missing/incompatible packages, run:

```powershell
python -m pip install -e ".[dev]"
```

Then:

```powershell
python 00_setup/02_setup_check.py --quick
python 00_setup/07_prepare_if_needed.py
python -m streamlit run 08_ui/app.py
```

No unconditional `pip --upgrade` step is required.

Optional `.env` values are documented in `.env.example`.

`TRAVEL_DATA_MODE=local` gives deterministic offline execution.

`TRAVEL_DATA_MODE=auto` tries live weather/FX providers and falls back to local data.

`TRAVEL_DATA_MODE=live` requires live provider access.


## Visualization dependency

The Project Statistics UI uses `plotly>=6,<7` for interactive charts. The normal setup dependency check treats Plotly exactly like the other required packages: it installs it only when missing or incompatible.
