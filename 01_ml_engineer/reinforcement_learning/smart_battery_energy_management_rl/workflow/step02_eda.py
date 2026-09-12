from __future__ import annotations
import matplotlib.pyplot as plt
from workflow.shared.utils import ROOT, ensure_data, section, save_json


def main(config_path=None, quick=False):
    section("STEP 02 — Energy data analysis")
    full, train, val, test = ensure_data(config_path)
    metrics = {
        "price_mean": float(full.electricity_price_eur_per_kwh.mean()),
        "price_p95": float(full.electricity_price_eur_per_kwh.quantile(0.95)),
        "demand_mean_kw": float(full.electricity_demand_kw.mean()),
        "demand_peak_kw": float(full.electricity_demand_kw.max()),
        "renewable_mean_kw": float(full.renewable_generation_kw.mean()),
        "renewable_share_proxy_pct": float(100 * full.renewable_generation_kw.sum() / full.electricity_demand_kw.sum()),
    }
    save_json(metrics, "04_results/metrics/step02_eda_summary.json")

    hourly = full.groupby("hour_of_day")[["electricity_price_eur_per_kwh","electricity_demand_kw","renewable_generation_kw"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5)); hourly.plot(ax=ax); ax.set_title("Average daily profiles"); ax.set_xlabel("Hour of day"); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(ROOT/"04_results/figures/04_average_daily_profiles.png", dpi=160); plt.close(fig)
    corr = full[["electricity_price_eur_per_kwh","electricity_demand_kw","renewable_generation_kw"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5)); im=ax.imshow(corr, vmin=-1, vmax=1); ax.set_xticks(range(3), ["price","demand","renewable"], rotation=20); ax.set_yticks(range(3), ["price","demand","renewable"]); fig.colorbar(im, ax=ax); ax.set_title("Feature correlation"); fig.tight_layout(); fig.savefig(ROOT/"04_results/figures/05_feature_correlation.png", dpi=160); plt.close(fig)
    print(metrics)
    return metrics

if __name__ == "__main__": main()
