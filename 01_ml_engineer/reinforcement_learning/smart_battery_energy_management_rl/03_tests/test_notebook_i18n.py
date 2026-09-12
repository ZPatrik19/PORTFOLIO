from __future__ import annotations

import json
from pathlib import Path
import shutil

from tools.set_notebook_language import switch_notebook

ROOT = Path(__file__).resolve().parents[1]


def _embedded_png_count(nb: dict) -> int:
    count = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            count += int("image/png" in output.get("data", {}))
    return count


def test_all_markdown_cells_have_hu_and_en_variants():
    notebooks = sorted((ROOT / "workflow").glob("step*.ipynb"))
    assert notebooks
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        markdown_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
        assert markdown_cells
        for cell in markdown_cells:
            variants = cell.get("metadata", {}).get("i18n", {})
            assert variants.get("hu")
            assert variants.get("en")


def test_language_switch_preserves_outputs_and_code(tmp_path: Path):
    source = ROOT / "workflow" / "step03_environment_design.ipynb"
    target = tmp_path / source.name
    shutil.copy2(source, target)

    before = json.loads(target.read_text(encoding="utf-8"))
    before_code = [c["source"] for c in before["cells"] if c["cell_type"] == "code"]
    before_png = _embedded_png_count(before)

    switch_notebook(target, "en")
    switch_notebook(target, "hu")

    after = json.loads(target.read_text(encoding="utf-8"))
    after_code = [c["source"] for c in after["cells"] if c["cell_type"] == "code"]
    after_png = _embedded_png_count(after)

    assert before_code == after_code
    assert before_png == after_png
    assert after["metadata"]["project_language"] == "hu"
    assert after["metadata"]["kernelspec"]["name"] == "battery-energy-rl"
