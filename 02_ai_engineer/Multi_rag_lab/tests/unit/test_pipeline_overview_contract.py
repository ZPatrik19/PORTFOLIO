from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_overview_is_registered_in_navigation() -> None:
    app = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    page = (ROOT / "ui" / "pages" / "pipeline_overview.py").read_text(encoding="utf-8")
    assert '"Pipeline térkép"' in app
    assert '"pipeline_overview"' in app
    assert "Offline / Indexing pipeline" in page
    assert "Online / Query + Answering pipeline" in page
    assert "Evaluation / Performance / Operations" in page


def test_pipeline_benchmark_has_in_ui_asset_repair() -> None:
    page = (ROOT / "ui" / "pages" / "pipeline_matrix.py").read_text(encoding="utf-8")
    assert "Benchmark előkészítése" in page
    assert "100 cikkes korpusz + evaluation dataset előkészítése" in page
    assert "Evaluation dataset újraépítése" in page
