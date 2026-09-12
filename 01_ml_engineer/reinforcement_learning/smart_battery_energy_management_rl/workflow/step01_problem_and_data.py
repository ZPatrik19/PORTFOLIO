from __future__ import annotations
from workflow.shared.utils import ROOT, ensure_data, load_config, section, save_json
from workflow.shared.visualization import plot_timeseries


def main(config_path=None, quick=False):
    section("STEP 01 — Problem definition + synthetic energy data")
    cfg = load_config(config_path)
    full, train, val, test = ensure_data(config_path)
    first = full.head(24 * 7)
    figdir = ROOT / "04_results/figures"
    plot_timeseries(first, "timestamp", "electricity_price_eur_per_kwh", "Electricity price — first 7 days", "EUR/kWh", figdir / "01_electricity_price.png")
    plot_timeseries(first, "timestamp", "electricity_demand_kw", "Electricity demand — first 7 days", "kW", figdir / "02_demand.png")
    plot_timeseries(first, "timestamp", "renewable_generation_kw", "Renewable generation — first 7 days", "kW", figdir / "03_renewable_generation.png")
    split_info = {
        "rows_total": len(full), "rows_train": len(train), "rows_validation": len(val), "rows_test": len(test),
        "train_start": str(train.timestamp.min()), "train_end": str(train.timestamp.max()),
        "validation_start": str(val.timestamp.min()), "validation_end": str(val.timestamp.max()),
        "test_start": str(test.timestamp.min()), "test_end": str(test.timestamp.max()),
        "note": "Chronological split: validation/test remain in the future relative to training."
    }
    save_json(split_info, "04_results/metrics/step01_data_split.json")
    print(split_info)
    return split_info

if __name__ == "__main__": main()
