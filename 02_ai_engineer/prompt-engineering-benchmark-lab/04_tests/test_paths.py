"""EN: Portable repository-root discovery and environment override behavior.

HU: A hordozható repository-root felderítést és az environment override működését ellenőrzi.
"""

from prompt_benchmark.paths import PATHS


def test_project_paths_point_to_repository_layout() -> None:
    """EN: Verifies canonical ProjectPaths resolve to the expected repository directories.

    HU: Ellenőrzi, hogy a ProjectPaths a megfelelő repository mappákra mutat.
    """
    assert (PATHS.root / "pyproject.toml").exists()
    assert PATHS.configs.name == "configs"
    assert PATHS.results == PATHS.root / "07_outputs" / "results"
    assert PATHS.prompts == PATHS.root / "configs" / "prompts"


def test_project_root_env_override(monkeypatch, tmp_path):
    """EN: Verifies PROMPT_BENCHMARK_ROOT can explicitly select a valid runtime repository root.

    HU: Ellenőrzi, hogy a PROMPT_BENCHMARK_ROOT explicit runtime repository rootot tud kijelölni.
    """
    from prompt_benchmark.paths import ROOT_ENV_VAR, discover_project_root

    for marker in ("configs", "01_data", "05_scripts"):
        (tmp_path / marker).mkdir()
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    assert discover_project_root() == tmp_path.resolve()


def test_path_methods_are_called_after_path_composition() -> None:
    """EN: Prevents string-path method precedence bugs such as ``root / "file.csv".exists()``.

    HU: Megakadályozza az olyan string/path precedenciahibákat, mint a ``root / "file.csv".exists()``.
    """
    import ast

    path_methods = {
        "exists",
        "is_file",
        "is_dir",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "open",
        "stat",
        "unlink",
    }
    violations: list[str] = []
    scan_roots = [PATHS.root / "03_src", PATHS.root / "05_scripts"]

    for scan_root in scan_roots:
        for source_path in scan_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                    continue
                right = node.right
                if not (
                    isinstance(right, ast.Call)
                    and isinstance(right.func, ast.Attribute)
                    and right.func.attr in path_methods
                    and isinstance(right.func.value, ast.Constant)
                    and isinstance(right.func.value.value, str)
                ):
                    continue
                relative_path = source_path.relative_to(PATHS.root)
                violations.append(f"{relative_path}:{node.lineno} -> string.{right.func.attr}() before Path composition")

    assert not violations, "Invalid pathlib precedence found:\n" + "\n".join(violations)
