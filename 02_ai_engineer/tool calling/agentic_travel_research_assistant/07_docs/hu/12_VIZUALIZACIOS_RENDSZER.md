# Vizualizációs rendszer és dashboard-stílus

## Bevezetés

A projekt három analitikai felülete — **Project Statistics**, **Data Quality** és **Live Statistics** — közös vizualizációs rendszert használ. A cél nem pusztán az, hogy sok grafikon jelenjen meg, hanem hogy minden ábra ugyanazokat a vizuális szabályokat kövesse, jól olvasható legyen világos és sötét háttéren, valamint összehasonlítható méretben jelenjen meg.

A közös megvalósítás helye:

```text
08_ui/chart_theme.py
```

Ez a modul választja szét a chartok tartalmát a megjelenítési szabályoktól. A dashboard-modulok ezért az adat- és chartlogikára koncentrálnak, míg a színkontraszt, a magasság, a tengelyek és a hover-stílus központilag kezelhető.

## 1. Választható Light és Dark chart theme

A Streamlit sidebarban külön **Analytics chart theme** választó található:

- `Light · high contrast`
- `Dark · high contrast`

A chart theme független a routing módszertantól és az adat-provider módtól. A beállítás egyszerre érvényes mindhárom analitikai dashboard Plotly-ábráira.

### Light theme

Világos módban a vizuális alapelv:

```text
háttér: fehér
fő szöveg: közel fekete
axis/tick szöveg: közel fekete
grid: jól látható világosszürke
hover: fehér háttér + fekete szöveg
```

A fő szövegszín `#111111`, ezért nincs világosszürke szöveg fehér háttéren.

### Dark theme

Sötét módban:

```text
háttér: #0E1117
fő szöveg: fehér
axis/tick szöveg: fehér
grid: sötétszürke, de látható
hover: sötét háttér + fehér szöveg
```

Ez megakadályozza azt a helyzetet, amikor egy dark charton a Plotly alapértelmezett sötétszürke feliratai eltűnnek.

## 2. Egységes chartmagasság

Az interaktív dashboardok fő grafikonjai közös magasságot használnak:

```python
CHART_HEIGHT = 460
```

Ennek oka vizuális és funkcionális:

1. a kétoszlopos dashboard-sorok kártyái egy vonalban maradnak;
2. lapváltáskor nem változik szélsőségesen a layout magassága;
3. összehasonlító chartok — például radar és quality/latency scatter — azonos vizuális súlyt kapnak;
4. kisebb a „dashboard collage” hatás.

A chart-builder függvények adhatnak saját belső layoutot, de a végső render-réteg egységesíti a megjelenített magasságot.

## 3. Közös Plotly render pipeline

A dashboard chart útvonala:

```text
adat / CSV / SQLite
        ↓
chart builder
        ↓
Plotly Figure
        ↓
style_figure(...)
        ↓
Light vagy Dark kontraszt
+ fix height
+ axis styling
+ legend styling
+ hover styling
        ↓
render_plotly(...)
        ↓
Streamlit
```

A dashboardok nem közvetlenül `st.plotly_chart()`-ot hívnak. A közös `render_plotly()` wrapper biztosítja:

- a témát;
- a fix magasságot;
- a `width="stretch"` elrendezést;
- a Plotly responsive módot;
- az explicit Streamlit `key` használatát;
- a Streamlit saját Plotly theme-jének kikapcsolását (`theme=None`), hogy a projekt saját kontrasztszabályai érvényesüljenek.

## 4. Kontrasztszabályok

A közös stílusréteg explicit módon állítja:

- `paper_bgcolor`;
- `plot_bgcolor`;
- layout font color;
- title font color;
- x/y axis tick font;
- x/y axis title font;
- gridline szín;
- zero-line szín;
- axis-line szín;
- legend font;
- hover label background/text;
- annotation font;
- polar/radar axis szöveg;
- colorbar tick és title szöveg.

Ez azért fontos, mert a Plotly chartok különböző típusai nem ugyanabból az alapértelmezett fontbeállításból örökölnek mindent. Egy egyszerű `template="plotly_dark"` önmagában nem garantálná, hogy minden annotation, radar label vagy colorbar azonos kontrasztot használ.

## 5. Interaktív és mentett plotok

A projekt két vizualizációs formát tart fenn.

### Interaktív Plotly chart

A Streamlit dashboard elsődleges nézete. Támogatja:

- hover részleteket;
- zoomot;
- pan műveletet;
- legend ki-/bekapcsolást;
- tooltipet;
- dinamikus filtereket;
- responsive szélességet.

### Mentett statikus PNG

A `06_results/project_statistics/` és `06_results/data_quality/` mappában tárolt reprodukálható snapshotok. Ezek mindig magas kontrasztú **világos** témát használnak:

```text
fehér háttér + fekete szöveg
```

Ennek oka, hogy egy PNG nem tud futásidőben témát váltani. A fix light snapshot alkalmas GitHub README-ba, dokumentációba és exportált riportba is.

## 6. Három dashboard közös vizuális nyelve

### Project Statistics

A projekt statikus állapotát mutatja:

- dataset scale;
- memóriahasználat;
- query- és entity-diverzitás;
- benchmark-komplexitás;
- router precision/recall/F1;
- methodology quality/latency.

### Data Quality

Az adatok megbízhatóságát diagnosztizálja:

- quality gate-ek;
- query diversity;
- repetition risk;
- entity-name diversity;
- split leakage;
- intent balance.

### Live Statistics

A tényleges UI-használatból számol:

- kérdés/tool-call trend;
- run success;
- tool reliability;
- latency distribution;
- top workflow-k;
- destination mix;
- methodology usage;
- history.

A három dashboard tartalma eltér, de a színezés, fontkontraszt, magasság és interakciós modell közös.

## 7. Új chart hozzáadásának szabálya

Új interaktív chart hozzáadásakor:

1. a chart-builder csak `plotly.graph_objects.Figure` objektumot adjon vissza;
2. ne állítson alacsony kontrasztú fix szövegszínt;
3. a renderelés a közös `render_plotly()` wrapperen keresztül történjen;
4. minden Streamlit chart kapjon egyedi `key` értéket;
5. a dashboard renderfüggvény adja tovább a `chart_theme` paramétert;
6. a chart ne használja a deprecated `use_container_width` API-t;
7. az alap dashboard chart magassága maradjon `CHART_HEIGHT`.

## 8. Tesztelés

A vizualizációs réteghez külön regressziós tesztek tartoznak. Ellenőrzik többek között, hogy:

- Light módban a háttér fehér és a szöveg fekete;
- Dark módban a háttér sötét és a szöveg fehér;
- minden chart ugyanazt az alapmagasságot kapja;
- mindhárom dashboard elfogadja a `chart_theme` paramétert;
- nincs deprecated Streamlit width API;
- a Plotly elemek explicit egyedi key-t használnak.

## Összegzés

A vizualizációs réteg külön komponenssé választása biztosítja, hogy a projekt analitikai felületei ne csak funkcionálisan, hanem vizuálisan is egy rendszernek tűnjenek. A felhasználó Light és Dark módban is magas kontrasztot kap, az összehasonlító chartok egységes méretűek, a statikus snapshotok pedig dokumentációs célra is stabilan olvashatók.
