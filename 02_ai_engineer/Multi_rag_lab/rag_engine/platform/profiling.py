from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class Timing:
    elapsed_ms: float = 0.0


@contextmanager
def timer() -> Timing:
    result = Timing()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


def process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2
