from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_context_profile_uses_callback_without_post_widget_mutation_button() -> None:
    text = (ROOT / "ui/components/common.py").read_text(encoding="utf-8")
    assert "def _sync_context_budget_from_profile" in text
    assert "on_change=_sync_context_budget_from_profile" in text
    assert "Ajánlott tokenkeret alkalmazása" not in text


def test_ui_uses_current_width_api() -> None:
    offenders = []
    for path in (ROOT / "ui").rglob("*.py"):
        if "use_container_width=" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_removed_workflow_module_stays_removed() -> None:
    assert not (ROOT / "ui/components/workflow.py").exists()
