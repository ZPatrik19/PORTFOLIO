"""Validate repository structure, data scale, imports, registry and one multi-tool smoke run."""
from __future__ import annotations
import argparse,csv,importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
os.environ["TRAVEL_DATA_MODE"]="local"
REQUIRED=[
    "01_data/raw/cities.csv","01_data/raw/hotels.csv","01_data/raw/attractions.csv","01_data/raw/restaurants.csv","01_data/raw/transport.csv",
    "01_data/benchmark/agent_tasks.json","02_notebooks/01_data_exploration.ipynb","03_src/travel_agent/tools/registry.py",
    "04_scripts/01_run_demo.py","04_scripts/04_run_live_openai_agent.py","04_scripts/07_prepare_train_evaluate.py","04_scripts/08_validate_presets.py","04_scripts/09_generate_project_statistics.py","05_tests/test_tools.py","05_tests/test_usage_store.py",
    "07_docs/README.md","07_docs/hu/00_DOKUMENTACIOS_TERKEP.md","07_docs/hu/01_PROJEKT_ATTEKINTES.md","07_docs/hu/04_TOOL_CALLING_ES_TOOL_TERVEZES.md","07_docs/hu/07_EVALUATION_ES_STATISZTIKAK.md",
    "07_docs/en/00_DOCUMENTATION_INDEX.md","07_docs/en/01_PROJECT_OVERVIEW.md","07_docs/en/04_TOOL_CALLING_AND_TOOL_DESIGN.md","07_docs/en/07_EVALUATION_AND_STATISTICS.md","07_docs/shared/visuals/01_runtime.png",
    "03_src/travel_agent/presets.py","03_src/travel_agent/usage/store.py","08_ui/app.py","RUN_UI.bat","START_HERE_HU.md","START_HERE_EN.md","00_setup/06_check_dependencies.py","00_setup/07_prepare_if_needed.py",
]
def check(ok,label):
    print(f"[{'OK' if ok else 'FAIL'}] {label}")
    if not ok:raise SystemExit(1)
def count_csv(path):
    with path.open(encoding="utf-8",newline="") as f:return sum(1 for _ in csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true",help="Skip full data-quality audit and smoke workflows.")
    args=parser.parse_args()
    print("Agentic Travel Research Assistant — setup check\n")
    check(sys.version_info>=(3,10),f"Python {sys.version.split()[0]} >= 3.10")
    for rel in REQUIRED:check((ROOT/rel).exists(),rel)
    for package in ["pydantic","matplotlib","sklearn","joblib","pandas","streamlit"]:check(importlib.util.find_spec(package) is not None,f"dependency: {package}")
    check(count_csv(ROOT/"01_data/raw/hotels.csv")>=180000,"hotel dataset >= 180000 rows")
    check(count_csv(ROOT/"01_data/raw/attractions.csv")>=90000,"attraction dataset >= 90000 rows")
    check(count_csv(ROOT/"01_data/raw/restaurants.csv")>=90000,"restaurant dataset >= 90000 rows")
    check(count_csv(ROOT/"01_data/raw/intent_router_dataset.csv")>=240000,"intent-router dataset >= 240000 rows")
    from travel_agent.tools import ToolRegistry
    from travel_agent.presets import PRESET_QUESTIONS
    check(len(ToolRegistry().names)==8,"eight registered tools")
    check(len(PRESET_QUESTIONS)==30 and len({x["id"] for x in PRESET_QUESTIONS})==30,"30 unique bilingual preset scenarios")
    if args.quick:
        from travel_agent.training import model_status
        status=model_status()
        if status.get("exists") and not status.get("stale"):
            state="current"
        elif status.get("reason") == "runtime_version_mismatch":
            state="saved model uses a different sklearn/Python runtime — one-time local retraining will run next"
        elif status.get("reason") in {"model_metadata_missing", "model_load_failed"}:
            state="legacy/incompatible saved model — one-time local retraining will run next"
        else:
            state="missing/stale — preparation will run next"
        print(f"[INFO] ML router: {state}")
        print("\nQuick setup check is ready.")
        return

    from travel_agent.quality import audit_all
    quality=audit_all(write_outputs=False)
    check(all(quality["quality_gates"].values()),"all data-quality gates pass")
    check(quality["split_leakage"]["test__train"]["shared_patterns"]==0,"no normalized train/test template overlap")
    from travel_agent.agent import PlanThenExecuteTravelAgent
    q="I am going to Vienna for 3 days. Check weather, hotels under 150 EUR, museums and landmarks, restaurants under 35 EUR, and public transport."
    run=PlanThenExecuteTravelAgent().run(q)
    check(set(run.tool_names)=={"get_weather","search_hotels","search_attractions","search_restaurants","get_transport_options"},"English multi-tool smoke run")
    check(all(c.success for c in run.trace),"all English smoke-run tools succeeded")

    # Regression check for Hungarian morphology and amount-first price constraints.
    q_hu=(
        "3 napra megyek Bécsbe. Nézd meg az időjárást, keress 150 euró alatti hotelt, "
        "ajánlj éttermet és látnivalókat, valamint mondd meg a tömegközlekedési lehetőségeket."
    )
    run_hu=PlanThenExecuteTravelAgent(language="hu").run(q_hu)
    check(
        run_hu.tool_names==["get_weather","search_hotels","search_attractions","search_restaurants","get_transport_options"],
        "Hungarian declined-city multi-tool smoke run",
    )
    check(run_hu.trace[1].arguments.get("max_price_per_night_eur")==150.0,"Hungarian amount-first hotel budget parsing")
    check(all(c.success for c in run_hu.trace),"all Hungarian smoke-run tools succeeded")
    print("\nSetup is ready.")
if __name__=="__main__":main()
