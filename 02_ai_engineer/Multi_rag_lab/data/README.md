# Data

A `data/` könyvtár a pipeline bemeneteit tartalmazza.

- `demo/`: kis, verziózott mintaadat a gyors kipróbáláshoz.
- `raw/`: letöltött nyers dokumentumok; újragenerálható, ezért nincs Gitben tárolva.
- `processed/`: parsing/cleaning/chunking után előálló adatok; újragenerálható.

A magyar orvosi korpusz előkészítése:

```bash
python scripts/prepare_medical_corpus.py
```

A teljes corpus és a feldolgozott chunkok mérete, tartalma és létrehozási ideje környezetfüggő, ezért nem release-artifactként kerülnek a repositoryba.
