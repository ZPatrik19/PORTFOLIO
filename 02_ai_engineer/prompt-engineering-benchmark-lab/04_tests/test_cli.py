"""EN: Installed CLI argument forwarding and user-facing command validation.

HU: A telepített CLI argumentumtovábbítását és a felhasználói parancsvalidációt ellenőrzi.
"""

from __future__ import annotations

import pytest

from prompt_benchmark import cli


def test_cli_forwards_workflow_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """EN: Verifies that CLI flags such as provider, strategy, and limit are forwarded unchanged to workflow scripts.

    HU: Ellenőrzi, hogy a CLI a provider/strategy/limit és más workflow argumentumokat változtatás nélkül továbbítja.
    """
    captured: dict[str, object] = {}

    def fake_run(name: str, extra_args: list[str]) -> int:
        captured["name"] = name
        captured["args"] = extra_args
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_run)
    with pytest.raises(SystemExit) as exc:
        cli.main(["benchmark", "--provider", "mock", "--strategy", "p0_zero_shot", "--limit", "3"])
    assert exc.value.code == 0
    assert captured == {
        "name": "benchmark",
        "args": ["--provider", "mock", "--strategy", "p0_zero_shot", "--limit", "3"],
    }


def test_cli_rejects_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    """EN: Ensures unknown CLI commands fail with a non-zero exit code rather than being silently ignored.

    HU: Biztosítja, hogy az ismeretlen CLI parancs nem nulla exit kóddal álljon le.
    """
    with pytest.raises(SystemExit) as exc:
        cli.main(["does-not-exist"])
    assert exc.value.code == 2
    assert "Unknown command" in capsys.readouterr().err
