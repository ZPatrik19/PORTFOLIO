from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.ingestion.catalog import default_sources, download_source, load_source_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Magyar nyilvános RAG mintakorpusz letöltése.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_sources.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "hungarian_corpus")
    parser.add_argument("--all", action="store_true", help="Minden konfigurált forrás letöltése.")
    parser.add_argument("--ids", nargs="*", default=[], help="Csak a megadott source ID-k letöltése.")
    args = parser.parse_args()

    catalog = load_source_catalog(args.config)
    if args.ids:
        requested = set(args.ids)
        sources = [source for source in catalog if source.id in requested]
        missing = requested - {source.id for source in sources}
        if missing:
            raise SystemExit(f"Ismeretlen source ID: {', '.join(sorted(missing))}")
    elif args.all:
        sources = catalog
    else:
        sources = default_sources(args.config)

    if not sources:
        print("Nincs letöltendő forrás a config/data_sources.yaml fájlban.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.title}")
        try:
            path = download_source(source, args.out)
            print(f"  OK -> {path}")
        except Exception as exc:
            failures += 1
            print(f"  HIBA -> {exc}")
    print(f"Kész. Sikeres: {len(sources) - failures}, hibás: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
