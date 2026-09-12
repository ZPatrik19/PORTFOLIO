# STEP 00 Setup and Notebook Language Switching

## One-command setup

Run from the repository root:

```bash
python 00_setup_project.py
```

The script creates/reuses `.venv`, upgrades packaging tools, installs `.[dev]`, prepares the configured dataset, registers the `Python (battery-energy-rl)` Jupyter kernel, and runs a BatteryEnvironment smoke test.

Optional flags:

```bash
python 00_setup_project.py --recreate
python 00_setup_project.py --force-data
python 00_setup_project.py --with-sb3
python 00_setup_project.py --with-citylearn
python 00_setup_project.py --skip-kernel
python 00_setup_project.py --skip-smoke-test
```

The default dataset is synthetic, therefore it is generated locally and no download is needed. `workflow.shared.utils.ensure_data()` also supports `data.source: url` in `config.yaml` for future real-data experiments.

## Notebook language

The nine workflow notebooks store Hungarian and English markdown under `cell.metadata.i18n`.

```bash
python switch_language.py --language hu
python switch_language.py --language en
python switch_language.py --status
```

The operation changes only the active cell source. Existing code, execution counts, model results and embedded figures remain untouched.

Generate both language versions as derived copies:

```bash
python tools/set_notebook_language.py --export-both notebook_exports
```

`workflow/` remains the single source of truth; the export folders are optional generated artifacts.
