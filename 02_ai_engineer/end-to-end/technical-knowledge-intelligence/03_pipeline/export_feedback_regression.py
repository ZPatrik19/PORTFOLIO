from __future__ import annotations

from tkip.config import load_config
from tkip.feedback import export_negative_feedback


def main() -> int:
    """Export negative user feedback into regression samples."""
    samples = export_negative_feedback(load_config())
    print(f"Exported {len(samples)} negative-feedback regression samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
