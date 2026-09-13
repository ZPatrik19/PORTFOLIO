from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW = PROJECT_ROOT / "01_data" / "raw"
BENCH = PROJECT_ROOT / "01_data" / "benchmark"
RESULTS = PROJECT_ROOT / "06_results" / "data_quality"


def dataset_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _city_pattern() -> re.Pattern[str]:
    cities = pd.read_csv(RAW / "cities.csv")["city"].astype(str).tolist()
    aliases = cities + ["Bécs", "Prága", "Párizs", "Róma", "München", "Krakkó"]
    return re.compile("|".join(re.escape(x.casefold()) for x in sorted(set(aliases), key=len, reverse=True)))


def normalize_query(text: str, city_pattern: re.Pattern[str] | None = None) -> str:
    city_pattern = city_pattern or _city_pattern()
    s = str(text).casefold()
    s = city_pattern.sub("<city>", s)
    # Hungarian case suffixes attached to city placeholders are collapsed too.
    s = re.sub(r"<city>(?:ba|be|ban|ben|ból|ből|ra|re|ról|ről|on|en|ön|nál|nél|hoz|hez|höz)?", "<city>", s)
    s = re.sub(r"\b\d+(?:[.,]\d+)?\b", "<num>", s)
    s = re.sub(r"\b(?:eur|huf|usd|gbp|czk|pln|chf|sek|nok|dkk|ron|try|euro|euró|forint)\b|[€$£]", "<cur>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_metrics(df: pd.DataFrame, query_col: str = "query") -> dict[str, Any]:
    city_pattern = _city_pattern()
    normalized = df[query_col].astype(str).map(lambda x: normalize_query(x, city_pattern))
    counts = normalized.value_counts()
    n = len(df)
    return {
        "rows": n,
        "exact_duplicate_rows": int(df[query_col].duplicated().sum()),
        "unique_queries": int(df[query_col].nunique()),
        "normalized_patterns": int(normalized.nunique()),
        "normalized_pattern_ratio": float(normalized.nunique() / n) if n else 0.0,
        "largest_pattern_share": float(counts.iloc[0] / n) if n and len(counts) else 0.0,
        "median_pattern_frequency": float(counts.median()) if len(counts) else 0.0,
        "top_patterns": [{"pattern": str(k), "count": int(v)} for k, v in counts.head(12).items()],
    }


def _split_leakage(df: pd.DataFrame) -> dict[str, Any]:
    if "split" not in df.columns:
        return {}
    city_pattern = _city_pattern()
    work = df[["split", "query"]].copy()
    work["normalized"] = work["query"].map(lambda x: normalize_query(x, city_pattern))
    groups = {s: set(g["normalized"]) for s, g in work.groupby("split")}
    pairs: dict[str, Any] = {}
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = groups[a] & groups[b]
            denom = min(len(groups[a]), len(groups[b])) or 1
            pairs[f"{a}__{b}"] = {
                "shared_patterns": len(inter),
                "overlap_over_smaller_split": len(inter) / denom,
            }
    return pairs


def _entity_metrics(df: pd.DataFrame, name_col: str = "name") -> dict[str, Any]:
    names = df[name_col].astype(str)
    # Remove city names, digits and generic entity words to expose naming-template diversity.
    city_pattern = _city_pattern()
    skeleton = names.str.casefold().map(lambda x: city_pattern.sub("<city>", x))
    skeleton = skeleton.str.replace(r"\d+", "<num>", regex=True)
    generic = r"\b(?:hotel|residence|suites?|rooms?|house|stay|inn|lodge|apartments?|hostel|restaurant|kitchen|museum|landmark|gallery|park|market)\b"
    skeleton = skeleton.str.replace(generic, "<type>", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    counts = skeleton.value_counts()
    return {
        "rows": int(len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_names": int(names.duplicated().sum()),
        "unique_names": int(names.nunique()),
        "name_unique_ratio": float(names.nunique() / len(df)) if len(df) else 0.0,
        "name_skeleton_patterns": int(skeleton.nunique()),
        "name_skeleton_ratio": float(skeleton.nunique() / len(df)) if len(df) else 0.0,
        "largest_name_skeleton_share": float(counts.iloc[0] / len(df)) if len(df) and len(counts) else 0.0,
    }


def audit_all(write_outputs: bool = True) -> dict[str, Any]:
    query_files = ["sample_user_queries.csv", "intent_router_dataset.csv", "intent_router_challenge.csv"]
    entity_files = ["hotels.csv", "attractions.csv", "restaurants.csv"]
    report: dict[str, Any] = {"query_datasets": {}, "entity_datasets": {}, "split_leakage": {}, "files": {}}

    for name in query_files:
        path = RAW / name
        df = pd.read_csv(path)
        report["query_datasets"][name] = _query_metrics(df)
        report["files"][name] = {"sha256": dataset_hash(path), "bytes": path.stat().st_size}
        if name == "intent_router_dataset.csv":
            report["split_leakage"] = _split_leakage(df)
            if "tool_labels" in df.columns:
                counter: Counter[str] = Counter()
                for labels in df["tool_labels"].astype(str):
                    counter.update(x for x in labels.split("|") if x)
                report["router_label_counts"] = dict(sorted(counter.items()))

    for name in entity_files:
        path = RAW / name
        df = pd.read_csv(path)
        report["entity_datasets"][name] = _entity_metrics(df)
        report["files"][name] = {"sha256": dataset_hash(path), "bytes": path.stat().st_size}

    # Simple quality gates. These are intentionally explicit and inspectable.
    router = report["query_datasets"].get("intent_router_dataset.csv", {})
    sample = report["query_datasets"].get("sample_user_queries.csv", {})
    leakage = report.get("split_leakage", {})
    max_leak = max((v["overlap_over_smaller_split"] for v in leakage.values()), default=0.0)
    report["quality_gates"] = {
        "router_normalized_patterns_ge_1500": router.get("normalized_patterns", 0) >= 1500,
        "sample_normalized_patterns_ge_750": sample.get("normalized_patterns", 0) >= 750,
        "split_pattern_overlap_le_0_02": max_leak <= 0.02,
        "hotel_name_skeleton_ratio_ge_0_10": report["entity_datasets"].get("hotels.csv", {}).get("name_skeleton_ratio", 0) >= 0.10,
        "restaurant_name_skeleton_ratio_ge_0_10": report["entity_datasets"].get("restaurants.csv", {}).get("name_skeleton_ratio", 0) >= 0.10,
        "attraction_name_skeleton_ratio_ge_0_10": report["entity_datasets"].get("attractions.csv", {}).get("name_skeleton_ratio", 0) >= 0.10,
    }
    report["quality_gate_pass_rate"] = sum(report["quality_gates"].values()) / max(1, len(report["quality_gates"]))

    if write_outputs:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "data_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        for section in ("query_datasets", "entity_datasets"):
            for dataset, metrics in report[section].items():
                for metric, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        rows.append({"section": section, "dataset": dataset, "metric": metric, "value": value})
        pd.DataFrame(rows).to_csv(RESULTS / "data_quality_summary.csv", index=False)
        _write_charts(report)
    return report


def _write_charts(report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ["#2563EB", "#14B8A6", "#8B5CF6", "#F59E0B", "#EF4444"]

    q = report["query_datasets"]
    labels = [x.replace(".csv", "").replace("_", " ").title() for x in q]
    values = [q[x]["normalized_pattern_ratio"] for x in q]
    counts = [q[x]["normalized_patterns"] for x in q]
    fig, ax = plt.subplots(figsize=(12, 7.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(axis="both", colors="#111111", labelcolor="#111111")
    ax.xaxis.label.set_color("#111111"); ax.yaxis.label.set_color("#111111"); ax.title.set_color("#111111")
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Query corpus linguistic diversity", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("Unique normalized pattern ratio")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, ratio, count in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio + 0.025, f"{ratio:.1%}\n{count:,} patterns", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "query_pattern_diversity.png", dpi=190, facecolor="#FFFFFF")
    plt.close(fig)

    e = report["entity_datasets"]
    labels = [x.replace(".csv", "").replace("_", " ").title() for x in e]
    unique_ratio = [e[x]["name_unique_ratio"] for x in e]
    skeleton_ratio = [e[x]["name_skeleton_ratio"] for x in e]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 7.2), facecolor="#FFFFFF")
    b1 = ax.bar([i - width / 2 for i in x], unique_ratio, width, label="Unique-name ratio", color="#2563EB")
    b2 = ax.bar([i + width / 2 for i in x], skeleton_ratio, width, label="Name-skeleton ratio", color="#8B5CF6")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Entity naming diversity", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("Diversity ratio")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "entity_name_diversity.png", dpi=190, facecolor="#FFFFFF")
    plt.close(fig)

