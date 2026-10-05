from __future__ import annotations

import scripts.infrastructure_cli as infra


def test_cuda_wheel_channel_selection_for_python_314(monkeypatch) -> None:
    monkeypatch.setattr(infra, "_driver_cuda_version", lambda: (13, 0))
    assert infra._pytorch_cuda_install_spec() == (
        "https://download.pytorch.org/whl/cu126",
        "torch==2.14.1+cu126",
    )
    monkeypatch.setattr(infra, "_driver_cuda_version", lambda: (12, 8))
    assert infra._pytorch_cuda_index_url().endswith("/cu126")
    monkeypatch.setattr(infra, "_driver_cuda_version", lambda: (12, 6))
    assert infra._pytorch_cuda_index_url().endswith("/cu126")
    monkeypatch.setattr(infra, "_driver_cuda_version", lambda: (12, 5))
    assert infra._pytorch_cuda_install_spec() is None


def test_cuda_install_force_replaces_cpu_torch(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        infra,
        "_pytorch_cuda_install_spec",
        lambda: ("https://download.pytorch.org/whl/cu126", "torch==2.14.1+cu126"),
    )
    monkeypatch.setattr(infra, "run_stream", lambda cmd, **_: commands.append(list(cmd)) or 0)
    monkeypatch.setattr(infra, "show_cuda", lambda: True)
    assert infra.install_cuda_pytorch() is True
    flattened = commands[0]
    assert "--force-reinstall" in flattened
    assert "--no-cache-dir" in flattened
    assert "torch==2.14.1+cu126" in flattened
    assert "https://download.pytorch.org/whl/cu126" in flattened
