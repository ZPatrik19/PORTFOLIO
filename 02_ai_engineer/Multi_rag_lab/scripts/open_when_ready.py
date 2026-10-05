from __future__ import annotations

import sys
import time
import webbrowser
from urllib.parse import urlparse

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
parsed = urlparse(url)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(f"Unsupported health-check URL: {url}")

for _ in range(120):
    try:
        with requests.get(url, timeout=1) as response:
            if response.status_code < 500:
                webbrowser.open(url)
                raise SystemExit(0)
    except requests.RequestException:
        time.sleep(0.5)
raise SystemExit(1)
