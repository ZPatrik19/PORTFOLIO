"""Launch the local FastAPI and Streamlit services reliably.

The Windows batch files delegate process creation to this module instead of
nesting ``start`` + ``cmd`` + ``call`` commands.  This avoids the quoting and
label-resolution problems that often appear in paths containing spaces.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_HOST = "127.0.0.1"
API_PORT = 8000
UI_HOST = "127.0.0.1"
UI_PORT = 8501


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pids_on_port(port: int) -> list[int]:
    """Return local process ids listening on a port without extra dependencies."""
    pids: set[int] = set()
    if os.name == "nt":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "TCP":
                continue
            local = parts[1]
            state = parts[3].upper()
            if local.rsplit(":", 1)[-1] == str(port) and state == "LISTENING":
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    continue
    else:
        result = subprocess.run(
            ["bash", "-lc", f"command -v lsof >/dev/null && lsof -tiTCP:{port} -sTCP:LISTEN || true"],
            capture_output=True,
            text=True,
            check=False,
        )
        for value in result.stdout.split():
            if value.isdigit():
                pids.add(int(value))
    return sorted(pids)


def _stop_port_owner(port: int, label: str) -> None:
    pids = _pids_on_port(port)
    if not pids:
        return
    print(f"[INFO] Restarting {label}: stopping previous listener(s) on port {port}: {pids}")
    for pid in pids:
        if pid == os.getpid():
            continue
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                continue
    deadline = time.time() + 6
    while time.time() < deadline and _port_open("127.0.0.1", port):
        time.sleep(0.25)


def _creation_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_CONSOLE}
    return {"start_new_session": True}


def _spawn(command: list[str], label: str) -> subprocess.Popen:
    print(f"[START] {label}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        **_creation_kwargs(),
    )


def _wait_for_port(host: str, port: int, process: subprocess.Popen | None, label: str, timeout: int = 35) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(host, port):
            print(f"[OK] {label} is listening on http://{host}:{port}")
            return True
        if process is not None and process.poll() is not None:
            print(f"[ERROR] {label} exited early with code {process.returncode}.")
            return False
        time.sleep(0.5)
    print(f"[ERROR] {label} did not become ready within {timeout} seconds.")
    return False


def launch(*, api: bool = True, ui: bool = True, open_browser: bool = True, restart_existing: bool = True) -> int:
    processes: list[tuple[str, subprocess.Popen | None]] = []

    if api:
        if restart_existing and _port_open(API_HOST, API_PORT):
            _stop_port_owner(API_PORT, "FastAPI")
        if _port_open(API_HOST, API_PORT):
            print(f"[INFO] Reusing FastAPI on http://{API_HOST}:{API_PORT}")
            api_process = None
        else:
            api_process = _spawn(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "04_api.main:app",
                    "--host",
                    API_HOST,
                    "--port",
                    str(API_PORT),
                ],
                "FastAPI",
            )
        processes.append(("FastAPI", api_process))

    if ui:
        if restart_existing and _port_open(UI_HOST, UI_PORT):
            _stop_port_owner(UI_PORT, "Streamlit")
        if _port_open(UI_HOST, UI_PORT):
            print(f"[INFO] Reusing Streamlit on http://{UI_HOST}:{UI_PORT}")
            ui_process = None
        else:
            ui_process = _spawn(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "05_ui/app.py",
                    "--server.address",
                    UI_HOST,
                    "--server.port",
                    str(UI_PORT),
                    "--server.headless",
                    "true",
                ],
                "Streamlit",
            )
        processes.append(("Streamlit", ui_process))

    ok = True
    for label, process in processes:
        host, port = (API_HOST, API_PORT) if label == "FastAPI" else (UI_HOST, UI_PORT)
        ok = _wait_for_port(host, port, process, label) and ok

    if ok and ui and open_browser:
        webbrowser.open(f"http://localhost:{UI_PORT}")

    if ok:
        print("\n[READY] Technical Knowledge Intelligence is running.")
        if api:
            print(f"        API: http://localhost:{API_PORT}")
        if ui:
            print(f"        UI : http://localhost:{UI_PORT}")
        return 0

    print("\n[ERROR] One or more services failed to start. Check the service console window for the traceback.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch TKI local services")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--api-only", action="store_true")
    mode.add_argument("--ui-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Do not restart services already listening on TKI ports")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return launch(
        api=not args.ui_only,
        ui=not args.api_only,
        open_browser=not args.no_browser,
        restart_existing=not args.reuse_existing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
