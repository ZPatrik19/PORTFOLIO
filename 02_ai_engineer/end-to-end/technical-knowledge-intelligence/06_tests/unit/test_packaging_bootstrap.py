"""Regression tests for the simplified package/bootstrap contract."""

from __future__ import annotations

from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_is_the_dependency_source_of_truth() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["build-system"]["build-backend"] == "setuptools.build_meta"
    assert any(item.startswith("setuptools>=") for item in config["build-system"]["requires"])
    assert any(item.startswith("wheel>=") for item in config["build-system"]["requires"])
    assert config["project"]["dependencies"]
    assert "dev" in config["project"]["optional-dependencies"]
    assert "full" in config["project"]["optional-dependencies"]


def test_only_one_requirements_file_is_kept_at_repository_root() -> None:
    requirement_files = sorted(path.name for path in PROJECT_ROOT.glob("requirements*.txt"))

    assert requirement_files == ["requirements.txt"]
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-e ." in requirements
    assert ".[dev]" not in requirements


def test_windows_runner_bootstraps_build_backend_before_editable_install() -> None:
    script = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")

    backend_install = script.index('pip install "setuptools>=80" "wheel>=0.45"')
    backend_check = script.index("import setuptools.build_meta, wheel")
    editable_install = script.index("pip install -e . --no-build-isolation")

    assert backend_install < backend_check < editable_install


def test_linux_runner_bootstraps_build_backend_before_editable_install() -> None:
    script = (PROJECT_ROOT / "run_project.sh").read_text(encoding="utf-8")

    backend_install = script.index("pip install 'setuptools>=80' 'wheel>=0.45'")
    backend_check = script.index("import setuptools.build_meta, wheel")
    editable_install = script.index("pip install -e . --no-build-isolation")

    assert backend_install < backend_check < editable_install


def test_root_runner_surface_is_intentionally_small() -> None:
    batch_files = sorted(path.name for path in PROJECT_ROOT.glob("*.bat"))
    shell_files = sorted(path.name for path in PROJECT_ROOT.glob("*.sh"))

    assert batch_files == ["run_project.bat", "setup.bat"]
    assert shell_files == ["run_project.sh", "setup.sh"]


def test_first_run_wrappers_run_setup_then_application() -> None:
    windows_setup = (PROJECT_ROOT / "setup.bat").read_text(encoding="utf-8")
    linux_setup = (PROJECT_ROOT / "setup.sh").read_text(encoding="utf-8")

    assert 'run_project.bat" setup' in windows_setup
    assert 'run_project.bat" run' in windows_setup
    assert "./run_project.sh setup" in linux_setup
    assert "./run_project.sh run" in linux_setup
    assert "pip install" not in windows_setup
    assert "pip install" not in linux_setup




def test_daily_run_starts_only_the_streamlit_ui() -> None:
    windows = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")
    linux = (PROJECT_ROOT / "run_project.sh").read_text(encoding="utf-8")
    launcher = (PROJECT_ROOT / "05_scripts" / "launch_app.py").read_text(encoding="utf-8")

    assert "launch_app.py --ui-only" in windows
    assert "launch_app.py --ui-only" in linux
    assert '"--server.headless"' in launcher
    assert '"true"' in launcher

def test_daily_run_uses_python_service_launcher() -> None:
    windows_runner = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")
    linux_runner = (PROJECT_ROOT / "run_project.sh").read_text(encoding="utf-8")

    assert '05_scripts\\launch_app.py' in windows_runner
    assert "05_scripts/launch_app.py" in linux_runner
    launcher = (PROJECT_ROOT / "05_scripts" / "launch_app.py").read_text(encoding="utf-8")
    assert "subprocess.Popen" in launcher
    assert "streamlit" in launcher
    assert "uvicorn" in launcher


def test_single_runner_handles_test_profiles_without_extra_test_wrappers() -> None:
    windows = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")
    linux = (PROJECT_ROOT / "run_project.sh").read_text(encoding="utf-8")

    assert '06_tests\\run_suite.py "%PROFILE%"' in windows
    assert '06_tests/run_suite.py "$PROFILE"' in linux
    assert not (PROJECT_ROOT / "run_tests.bat").exists()
    assert not (PROJECT_ROOT / "run_tests.sh").exists()


def test_runtime_dependencies_do_not_include_test_only_or_unused_packages() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = "\n".join(config["project"]["dependencies"]).lower()
    dev = "\n".join(config["project"]["optional-dependencies"]["dev"]).lower()

    assert "scikit-learn" not in runtime
    assert "pypdf" not in runtime
    assert "httpx" not in runtime
    assert "httpx" in dev


def test_normal_setup_installs_runtime_only_and_dev_tools_are_lazy() -> None:
    windows = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")
    linux = (PROJECT_ROOT / "run_project.sh").read_text(encoding="utf-8")

    assert 'pip install -e . --no-build-isolation' in windows
    assert 'pip install -e ".[dev]" --no-build-isolation' in windows
    assert "pip install -e . --no-build-isolation" in linux
    assert "pip install -e '.[dev]' --no-build-isolation" in linux
    assert windows.index('pip install -e . --no-build-isolation') < windows.index(':ensure_dev_environment')
    assert linux.index("pip install -e . --no-build-isolation") < linux.index('ensure_dev_environment()')


def test_pytest_imports_tkip_from_current_repository_source() -> None:
    import tkip

    source = Path(tkip.__file__).resolve()
    expected_root = (PROJECT_ROOT / "03_pipeline" / "tkip").resolve()

    assert source.is_relative_to(expected_root), (
        f"Tests imported tkip from {source}, expected current repository source under {expected_root}"
    )


def test_windows_batch_files_use_crlf_line_endings() -> None:
    """Windows CMD label calls require release-safe CRLF batch files."""

    for batch_path in sorted(PROJECT_ROOT.glob("*.bat")):
        raw = batch_path.read_bytes()
        assert b"\r\n" in raw, f"{batch_path.name} does not contain CRLF line endings"
        assert raw.count(b"\n") == raw.count(b"\r\n"), (
            f"{batch_path.name} contains bare LF line endings; Windows CALL :label can fail"
        )


def test_shell_scripts_use_lf_line_endings() -> None:
    """Unix launchers should not contain CRLF characters."""

    for shell_path in sorted(PROJECT_ROOT.glob("*.sh")):
        raw = shell_path.read_bytes()
        assert b"\r\n" not in raw, f"{shell_path.name} unexpectedly contains CRLF line endings"


def test_windows_runner_subroutine_and_goto_labels_exist() -> None:
    """Every internal CALL/GOTO target in the Windows runner must resolve to a label."""

    import re

    script = (PROJECT_ROOT / "run_project.bat").read_text(encoding="utf-8")
    labels = set(re.findall(r"(?im)^\s*:([A-Za-z0-9_-]+)\s*$", script))
    call_targets = set(re.findall(r"(?i)\bcall\s+:([A-Za-z0-9_-]+)", script))
    goto_targets = set(re.findall(r"(?i)\bgoto\s+([A-Za-z0-9_-]+)", script))
    referenced = call_targets | goto_targets

    assert "ensure_python" in labels
    assert referenced <= labels, f"Missing batch labels: {sorted(referenced - labels)}"


def test_gitattributes_preserves_platform_line_endings() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "*.bat text eol=crlf" in attributes
    assert "*.sh  text eol=lf" in attributes
