from __future__ import annotations
import hashlib, json, re, time, uuid
from pathlib import Path
from typing import Iterable

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stable_id(*parts: str, prefix: str = "id") -> str:
    raw = "|".join(parts).encode("utf-8", "ignore")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:16]}"

def detect_language(text: str) -> str:
    t = " " + text.lower() + " "
    hu_terms=[" az "," és "," hogy "," egy "," lehet "," vagy "," hol "," melyik "," magyarázd "," tanítsd "," alapján "," keres "," működés", " könyv"]
    en_terms=[" the "," and "," that "," with "," from "," is "," what "," how "," where "," explain "," find "]
    hu=sum(t.count(x) for x in hu_terms) + sum(ch in t for ch in "őű")*2 + sum(ch in t for ch in "áéíóöü")
    en=sum(t.count(x) for x in en_terms)
    return "hu" if hu > en else "en"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ž0-9_+#.-]+", text.lower())

def jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def now_ms() -> float:
    return time.perf_counter() * 1000

def ids() -> tuple[str, str]:
    return uuid.uuid4().hex, uuid.uuid4().hex

def batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i+n]
