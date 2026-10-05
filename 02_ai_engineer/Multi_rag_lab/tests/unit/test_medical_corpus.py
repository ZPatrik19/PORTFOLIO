from __future__ import annotations

from pathlib import Path

from rag_engine.ingestion.medical import discover_medical_articles, load_medical_corpus_config


ROOT = Path(__file__).resolve().parents[2]


def test_medical_catalog_discovers_only_article_pages_and_is_diverse() -> None:
    config = load_medical_corpus_config(ROOT / "config" / "medical_corpus.yaml")
    html = """
    <html><body>
      <a href="/egeszseg-a-z/a-a/asztma.html">Asztma</a>
      <a href="/egeszseg-a-z/a-a/anafilaxia.html">Anafilaxia</a>
      <a href="/egeszseg-a-z/c-cs/cukorbetegseg-diabetesz.html">Cukorbetegség (Diabétesz)</a>
      <a href="/egeszseg-a-z/m/magasvernyomas.html">Magasvérnyomás-betegség</a>
      <a href="/egeszseg-a-z/z-zs/zoldhalyog.html">Zöldhályog (Glaucoma)</a>
      <a href="/egeszseg-a-z/a-a.html">A-Á</a>
      <a href="/rolunk/egeszsegvonal/oldalterkep.html">Oldaltérkép</a>
      <a href="https://example.com/egeszseg-a-z/x/x.html">Külső oldal</a>
    </body></html>
    """
    articles = discover_medical_articles(config, target=5, html_text=html)
    assert len(articles) == 5
    assert {item.title for item in articles} >= {
        "Asztma",
        "Anafilaxia",
        "Cukorbetegség (Diabétesz)",
        "Magasvérnyomás-betegség",
        "Zöldhályog (Glaucoma)",
    }
    assert all("egeszsegvonal.gov.hu" in item.url for item in articles)
    assert all(item.filename.endswith(".html") for item in articles)


def test_prepare_medical_corpus_uses_resolved_embedding_device_source():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "scripts" / "prepare_medical_corpus.py").read_text(encoding="utf-8")
    assert "device=embedding_device" in source
    assert "device=args.embedding_device" not in source


def test_medical_download_uses_reserve_articles_after_transient_failure(monkeypatch, tmp_path: Path) -> None:
    from rag_engine.ingestion.medical import MedicalArticle, download_medical_corpus
    import rag_engine.ingestion.medical as medical

    config = load_medical_corpus_config(ROOT / "config" / "medical_corpus.yaml")
    articles = [
        MedicalArticle("Első", "https://egeszsegvonal.gov.hu/egeszseg-a-z/e/elso.html", "e"),
        MedicalArticle("Második", "https://egeszsegvonal.gov.hu/egeszseg-a-z/m/masodik.html", "m"),
        MedicalArticle("Tartalék", "https://egeszsegvonal.gov.hu/egeszseg-a-z/t/tartalek.html", "t"),
    ]
    monkeypatch.setattr(medical, "discover_medical_articles", lambda *a, **k: articles)

    calls = []

    def fake_download(url, destination_dir, *, filename, **kwargs):
        calls.append(url)
        if "elso.html" in url:
            raise RuntimeError("transient disconnect")
        target = destination_dir / filename
        target.write_text("<html><body>ok</body></html>", encoding="utf-8")
        return target

    monkeypatch.setattr(medical, "download_url", fake_download)
    manifest = download_medical_corpus(config, tmp_path, target=2)
    assert manifest["available_documents"] == 2
    assert len(manifest["failures"]) == 1
    assert len(calls) == 3
