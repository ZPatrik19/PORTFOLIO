from pathlib import Path

import pytest

from tkip.config import ConfigurationError, _discover_project_root, load_config, resolve_path


def test_load_config_validates_default_configuration() -> None:
    config = load_config()

    assert config["chunking"]["overlap"] < config["chunking"]["chunk_size"]
    assert config["retrieval"]["final_k"] >= 1
    assert Path(config["_project_root"]).exists()


def test_invalid_chunk_overlap_raises_configuration_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        """
chunking:
  strategy: fixed
  chunk_size: 500
  overlap: 500
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="overlap"):
        load_config(invalid_config)


def test_resolve_path_is_repository_relative() -> None:
    resolved = resolve_path("01_data/reference_docs")

    assert resolved.is_absolute()
    assert resolved.name == "reference_docs"


def test_default_library_paths_use_clear_collection_names() -> None:
    config = load_config()

    assert config["paths"]["user_library"] == "01_data/user_library"
    assert config["paths"]["reference_docs"] == "01_data/reference_docs"


def test_project_root_discovery_prefers_runtime_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.yaml").write_text("project: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TKI_PROJECT_ROOT", raising=False)

    assert _discover_project_root() == tmp_path.resolve()
