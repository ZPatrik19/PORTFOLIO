from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_batch_files_use_crlf_and_no_bom() -> None:
    batch_files = sorted(ROOT.rglob("*.bat"))
    assert batch_files, "No .bat files found"
    for path in batch_files:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM is not allowed in {path}"
        assert b"\r\n" in raw, f"Missing CRLF line endings in {path}"
        assert raw.replace(b"\r\n", b"").find(b"\n") == -1, f"Bare LF found in {path}"
