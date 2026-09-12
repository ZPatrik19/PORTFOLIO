from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dry_bean.inference import load_bundle, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry Bean inference from one JSON feature vector")
    parser.add_argument("--input", required=True, help="Path to JSON containing exactly the 16 raw features")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model, scaler, encoder = load_bundle()
    print(json.dumps(predict(payload, model, scaler, encoder), indent=2))


if __name__ == "__main__":
    main()
