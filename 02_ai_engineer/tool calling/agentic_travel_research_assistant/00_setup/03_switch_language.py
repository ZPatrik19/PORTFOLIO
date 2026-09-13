"""Persist the default user-facing language in .env without touching code."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"


def set_key(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch default answer language between Hungarian and English.")
    parser.add_argument("language", choices=["hu", "en"])
    args = parser.parse_args()
    current = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    ENV.write_text(set_key(current, "APP_LANGUAGE", args.language), encoding="utf-8")
    label = "Hungarian / magyar" if args.language == "hu" else "English / angol"
    print(f"APP_LANGUAGE={args.language} saved to {ENV.name}. Default answer language: {label}.")


if __name__ == "__main__":
    main()
