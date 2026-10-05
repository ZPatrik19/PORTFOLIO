from pathlib import Path


def test_streamlit_auto_navigation_and_watcher_disabled():
    text = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert "showSidebarNavigation = false" in text
    assert 'fileWatcherType = "none"' in text
