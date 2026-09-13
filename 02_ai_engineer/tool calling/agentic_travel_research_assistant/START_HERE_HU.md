# Indulás — Agentic Travel Research Assistant

A projektet most már nem kell terminálból használni. A terminálos scriptek megmaradtak reprodukálhatóságra és tesztelésre, de a normál használat a böngészős UI.

## Projektleírások

A részletes technikai projektleírás külön magyar és angol dokumentumban is megtalálható:

- `07_docs/hu/00_DOKUMENTACIOS_TERKEP.md` — a teljes magyar dokumentáció belépési pontja
- `07_docs/en/00_DOCUMENTATION_INDEX.md` — az angol dokumentáció belépési pontja
- `07_docs/shared/visuals/` — közös architektúra- és folyamatábrák

## 1. Első indítás Windowson

Csomagold ki a ZIP-et, majd a projekt gyökerében kattints duplán erre:

```text
SETUP_AND_START_UI.bat
```

Ez:

1. létrehozza a `.venv` környezetet, ha még nincs;
2. aktiválja;
3. ellenőrzi a már telepített csomagok verzióit; ha minden megfelel, **nem telepít és nem frissít újra semmit**;
4. csak hiányzó vagy inkompatibilis dependency esetén telepít;
5. lefuttatja a setup ellenőrzést;
6. ellenőrzi a mentett ML router dataset-hashét, és csak hiányzó/elavult modell esetén tanít újra;
7. elindítja a böngészős felületet.

Később már elég:

```text
RUN_UI.bat
```

Ha kézzel akarod indítani:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run 08_ui/app.py
```

## 2. Mit tud a UI?

### Chat

A Chat fülön 30 különböző, magyar/angol preset kérdésből is választhatsz kategória és nehézség szerint, **de teljesen saját kérdést is beírhatsz**. A felület közvetlenül a szövegmező felett megmutatja, milyen információkat érdemes megadni: város, időtartam, árkeret, értékelés, éttermi vagy látnivaló-preferencia. Például:

```text
3 napra megyek Bécsbe. Nézd meg az időjárást, keress 150 euró alatti hotelt,
ajánlj éttermet és látnivalókat, valamint mondd meg a tömegközlekedési lehetőségeket.
```

A bal oldalon kiválasztható:

- `plan_execute`
- `ml_router`
- `openai_direct`
- magyar / angol nyelv
- `local / auto / live` adatforrás mód
- `Világos / Sötét` magas kontrasztú analytics diagramtéma

A válasz alatt megjelenik a teljes tool trace: toolnév, argumentumok, output, latency és sikeresség.

### Tool Explorer

Nem kell többé JSON-t escape-elni PowerShellben. A toolokat normál űrlapból lehet meghívni: hotelkeresés, étteremkeresés, weather, attraction, transport, currency stb.

### Data Quality

Itt látható:

- pontos duplikációk száma;
- normalizált nyelvi minták száma;
- train/test sablonátfedés;
- entity-nevek változatossága;
- quality gate-ek állapota;
- generált auditgrafikonok.

### Train & Evaluate

Innen egy gombbal újratanítható az intent router. A UI figyeli a training dataset hashét is: ha az adat megváltozott a modell tréningje óta, `stale` állapotot jelez és újratanítást kér.

Ugyanitt 50–22500 benchmark eseten futtatható evaluation.

### Live Statistics

Minden sikeresen lefuttatott Chat kérdés és válasz egy helyi SQLite adatbázisba kerül:

```text
06_results/usage/usage_history.sqlite3
```

A statisztika **valódi használatból változik**. Megmutatja többek között a kérdések számát, összes tool-hívást, átlagos tool/kérdés értéket, tool sikerarányt, átlagos latencyt, saját vs preset kérdéseket, módszertan-megoszlást, tool-gyakoriságot, napi használatot és a legutóbbi kérdés/válasz előzményeket. A history CSV-be exportálható és a UI-ból törölhető. Az adatbázis lokális, a projekt nem küldi külső szolgáltatáshoz.

### Project Statistics

Itt a repositoryból újraszámolt mérőszámok egy interaktív Plotly dashboardon láthatók: datasetméretek és memória, nyelvi/entity diverzitás, routing- és benchmark-komplexitás, router precision/recall/F1, methodology quality/latency trade-off és repository inventory. A grafikonok hoverrel, zoommal és szűrőkkel vizsgálhatók; külön **Saved Plots** galériában az előre generált PNG snapshotok is elérhetők. A statisztika egy gombbal újragenerálható.

### Dataset Explorer

Itt böngészhető a hotel-, attraction-, restaurant-, router-, challenge-, weather- és egyéb dataset. Város szerint is lehet szűrni.

## 3. Adatok

A korábbi nagy adathalmaz egyik problémája az volt, hogy sok sor valójában nagyon kevés sablonból készült. Ez torzíthatta a modellt és a teszteredményeket.

A javított corpus:

```text
240 000 router példa
├── 192 000 train
├──  24 000 validation
└──  24 000 held-out test

36 000 indirect/noisy challenge query
90 000 user-query sample
22 500 end-to-end agent benchmark
```

A jelenlegi auditban:

```text
exact duplicate router query:       0
normalizált router minták:          65 000+
train ↔ test normalizált overlap:   0
```

A hotel-, attraction- és restaurant-neveket is újrageneráltuk, hogy ne `Vienna Museum 001` / `Vienna Kitchen 002` jellegű ismétlődő minták domináljanak.

Az inventory továbbra is **szintetikus**, tehát nem valós foglalható hoteladatbázis. Ez tudatos: offline, determinisztikus tool- és agent-teszteléshez használjuk.

## 4. Modelltréning

A router már nem csak word TF-IDF-et használ:

```text
word TF-IDF
    +
Unicode accent normalization
    ↓
One-vs-Rest SGD logistic classifier
    ↓
labelenként optimalizált threshold
```

A karakter n-gramok segítenek például ilyen eseteknél:

```text
Bécsbe
Becsbe
esernyő
esernyo
közlekedés
kozlekedes
```

A threshold-választás precision-orientált, mert tool callingnál a fölösleges tool-hívás külön hibatípus.

A UI-ból kattints a **Train & Evaluate → Train / retrain intent router** gombra.

Terminálból ugyanaz:

```powershell
python 04_scripts/06_train_intent_router.py
```

## 5. Teljes automatikus pipeline

Ha mindent egyben szeretnél ellenőrizni:

```powershell
python 04_scripts/07_prepare_train_evaluate.py --limit 500
```

Ez sorrendben:

```text
Data-quality audit
        ↓
Quality gates
        ↓
Router training
        ↓
500-case agent evaluation
```

Ha az adatokat is újra akarod generálni:

```powershell
python 04_scripts/07_prepare_train_evaluate.py --regenerate-data --limit 500
```

## 6. Fő mappák

```text
00_setup      setup + adatgenerálás + minőségjavítás
01_data       raw / benchmark / processed adatok
02_notebooks  elemző notebookok
03_src        alkalmazáskód
04_scripts    CLI és pipeline scriptek
05_tests      automatizált tesztek
06_results    modellek, auditok, benchmark eredmények
07_docs       technikai dokumentáció
08_ui         Streamlit böngészős felület
```

A normál használathoz tehát a legegyszerűbb út:

```text
SETUP_AND_START_UI.bat
        ↓
Browser
        ↓
Train & Evaluate (ha szükséges)
        ↓
Chat
```

## 7. A 30 preset automatikus ellenőrzése

```powershell
python 04_scripts/08_validate_presets.py
```

Ez a 30 scenario magyar és angol változatát is lefuttatja `plan_execute` és `ml_router` módban, majd ellenőrzi az elvárt tool-route-ot.

## Modell-kompatibilitás

A router `joblib`/scikit-learn modell. Eltérő scikit-learn verzióból származó mentett modellek betöltése nem megbízható, ezért a projekt külön `intent_router_metadata.json` fájlban tárolja a tréning Python/scikit-learn verzióját és az adat hash-ét. A setup ezt **a modell betöltése előtt** ellenőrzi. Ha például a mentett modell sklearn 1.8 alatt készült, de nálad 1.9 fut, a setup nem downgrade-eli a csomagokat: egyszer újratanítja a routert a te környezetedben, majd a következő indításoknál kihagyja a tréninget.
