from __future__ import annotations

import sys
import time
import urllib.request
import webbrowser

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status < 500:
                webbrowser.open(url)
                raise SystemExit(0)
    except Exception:
        time.sleep(0.5)
raise SystemExit(1)
