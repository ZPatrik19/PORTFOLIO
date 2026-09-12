# Dry Bean Classification — Projekt dokumentáció

## Egy notebook-alapú ML feladattól egy reprodukálható, portfólió-szintű Machine Learning rendszerig

Ez a dokumentum azt mutatja be, hogy **mi készült el ténylegesen a projektben, miért születtek az egyes mérnöki döntések, hogyan fejlődött a rendszer, és hogyan érdemes a végleges repository-t egy recruiternek vagy ML Engineernek értelmeznie**.

A projekt a **UCI Dry Bean Dataset** adathalmazon old meg egy hétosztályos, tabuláris klasszifikációs problémát. A cél nem pusztán egy modell betanítása volt, hanem a teljes supervised Machine Learning életciklus bemutatása:

```text
adatbeszerzés
→ validáció
→ feltáró adatelemzés
→ leakage-safe adatszétválasztás
→ preprocessing
→ klasszikus modellek benchmarkja
→ training-data augmentation kísérlet
→ PyTorch modell tanítása
→ hiperparaméter-összehasonlítás
→ végső tesztértékelés
→ hibák és confidence elemzése
→ inference
→ monitoring és drift elemzés
→ reprodukálható riportolás
```

A projekt tudatosan úgy kezeli a modellezést, mint **egy nagyobb mérnöki workflow egyik lépését**.

---

# 1. Problémadefiníció

Az adathalmaz szárazbab-szemek képeiből származó morfológiai méréseket tartalmaz.

A feladat hét babfajta egyikének előrejelzése:

```text
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SEKER
SIRA
```

Minden megfigyelés 16 numerikus input feature-t tartalmaz, amelyek az alakot és geometriát írják le:

```text
Area
Perimeter
MajorAxisLength
MinorAxisLength
AspectRatio
Eccentricity
ConvexArea
EquivDiameter
Extent
Solidity
Roundness
Compactness
ShapeFactor1
ShapeFactor2
ShapeFactor3
ShapeFactor4
```

Ez egy többosztályos supervised classification probléma.

Az elsődleges minőségi metrika a **Macro F1**, mert így minden osztály azonos súllyal járul hozzá a végső eredményhez. Az accuracy szintén mérésre kerül, de nem használható arra, hogy elfedje egy gyengébben teljesítő osztály problémáit.

---

# 2. A projekt fő mérnöki célja

A repository egy központi elv köré épül:

> A notebook magyarázza a projektet, a `src/` tartalmazza a reusable implementációt, a workflow scriptek automatizálják az egyes lépéseket, a `run_project.py` pedig összeköti a teljes pipeline-t.

A projekt ezért kétféle használatot támogat.

### Interaktív / oktatási mód

A Jupyter notebookok bemutatják:

- mit csinálunk;
- miért szükséges;
- milyen alternatívák léteznek;
- hogyan néznek ki a köztes eredmények;
- hogyan értelmezzük az ábrákat;
- milyen következtetések adódnak az egyes kísérletekből.

### Reprodukálható / automatizált mód

Ugyanez a logika workflow scripteken és egy központi orchestratoron keresztül is futtatható.

A normál felhasználói workflow szándékosan egyszerű:

```bash
python 00_setup_project.py
python run_project.py
```

Az első parancs előkészíti a projektet.

A második lefuttatja a teljes ML workflow-t.

---

# 3. A végleges repository-struktúra

A projektet egy felülről lefelé követhető workflow-vá szerveztük át.

```text
dry-bean-classification/
│
├── 00_setup_project.py
├── 00_setup_project.ipynb
│
├── 01_data/
│   ├── raw/
│   └── processed/
│
├── workflow/
│   ├── step01_data_acquisition_validation.py
│   ├── step01_data_acquisition_validation.ipynb
│   ├── step02_eda.py
│   ├── step02_eda.ipynb
│   ├── step03_preprocessing.py
│   ├── step03_preprocessing.ipynb
│   ├── step04_baselines.py
│   ├── step04_baselines.ipynb
│   ├── step05_pytorch_training.py
│   ├── step05_pytorch_training.ipynb
│   ├── step06_evaluation_error_analysis.py
│   ├── step06_evaluation_error_analysis.ipynb
│   ├── step07_inference.py
│   ├── step07_inference.ipynb
│   ├── step08_monitoring_drift.py
│   ├── step08_monitoring_drift.ipynb
│   └── step09_generate_report.py
│
├── src/
│   └── dry_bean/
│       ├── augmentation.py
│       ├── benchmark.py
│       ├── config.py
│       ├── constants.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── models.py
│       ├── monitoring.py
│       ├── preprocessing.py
│       ├── training.py
│       ├── utils.py
│       ├── validation.py
│       └── visualization.py
│
├── 03_tests/
├── 04_results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── preexecuted_preview/
│
├── 05_graphviz/
├── tools/
├── reports/
│
├── config.yaml
├── predict.py
├── pyproject.toml
├── requirements.txt
├── run_project.py
├── switch_language.py
└── README.md
```

A struktúra szándékosan kerüli a felesleges nestinget. Ez portfólióprojekt, nem enterprise framework.

---

# 4. Futtatási workflow

A végső pipeline egyértelműen sorrendbe rendezett.

```text
STEP 00 — Project Setup
        ↓
STEP 01 — Data Acquisition & Validation
        ↓
STEP 02 — Exploratory Data Analysis
        ↓
STEP 03 — Split, Preprocessing & Augmentation
        ↓
STEP 04 — Classical Algorithm Benchmark
        ↓
STEP 05 — PyTorch Training & Tuning
        ↓
STEP 06 — Final Evaluation & Error Analysis
        ↓
STEP 07 — Inference
        ↓
STEP 08 — Monitoring & Drift
        ↓
STEP 09 — Report / Figure Generation
```

Ez megszünteti azt a bizonytalanságot, hogy mit kell előbb futtatni, és mely outputokra van szükség a későbbi lépésekhez.

---

# 5. STEP 00 — Automatikus project setup

A `00_setup_project.py` azért készült, hogy egy új fejlesztőnek ne kelljen manuálisan rekonstruálnia a Python környezetet.

A script feladata:

```text
.venv létrehozása
→ dependency-k telepítése
→ projekt telepítése pip install -e . módban
→ runtime mappák létrehozása
→ Jupyter kernel regisztrálása
→ dependency/import smoke test
→ dataset ellenőrzése
→ UCI adat letöltése, ha hiányzik
```

Normál setup:

```bash
python 00_setup_project.py
```

Environment újraépítése:

```bash
python 00_setup_project.py --recreate
```

Dataset kényszerített újraletöltése:

```bash
python 00_setup_project.py --force-data
```

A setup és a futtatás felelőssége szét van választva:

```text
00_setup_project.py = gép/projekt előkészítése
run_project.py      = teljes ML pipeline futtatása
```

---

# 6. STEP 01 — Adatbeszerzés és schema validation

A hivatalos adatforrás a UCI Machine Learning Repository 602-es datasetje.

A pipeline nem feltételezi automatikusan, hogy egy lemezen talált CSV megfelelő.

A szigorú data contract ellenőrzi:

- a várt 16 feature jelenlétét;
- a target oszlopot;
- a numerikus feature-típusokat;
- missing value-kat;
- duplikált sorokat;
- ismert target kategóriákat;
- váratlan oszlopokat;
- az adathalmaz dimenzióit.

A fejlesztés során különösen fontos volt, hogy akár egy apró néveltérés, például:

```text
AspectRation  vs  AspectRatio
roundness     vs  Roundness
```

is képes megtörni vagy észrevétlenül hibássá tenni a preprocessing pipeline-t.

A projekt ezért **fail-fast validation strategy**-t alkalmaz.

Hibás schema esetén a pipeline még a training előtt leáll.

---

# 7. STEP 02 — Exploratory Data Analysis

Az EDA jóval több lett egy egyszerű class-count ábránál.

A projekt vizsgálja:

- class distribution;
- numerikus feature-eloszlásokat;
- korrelációs struktúrát;
- outliereket;
- osztályonkénti boxplotokat;
- kiválasztott feature-pair scatter plotokat;
- PCA 2D projekciót;
- leíró statisztikákat.

Reprezentatív outputok:

```text
01_class_distribution.png
02_feature_distributions.png
03_correlation_heatmap.png
04_pca_2d.png
eda_boxplots_by_class.png
eda_scatter_1_Area_Perimeter.png
eda_scatter_2_Compactness_Eccentricity.png
```

Az EDA célja nem dekoratív vizualizáció.

Gyakorlati kérdésekre keres választ:

- Mennyire kiegyensúlyozottak az osztályok?
- Mely feature-ök erősen korreláltak?
- Vannak geometriailag könnyebben szeparálható osztályok?
- Nagyon eltérőek-e a feature scale-ek?
- Vannak-e nyilvánvaló eloszlási anomáliák?
- Indokolt lehet-e nemlineáris döntési határ?

---

# 8. STEP 03 — Adatszétválasztás és leakage-safe preprocessing

Az adathalmaz felosztása:

```text
70% train
15% validation
15% test
```

stratification mellett.

A három split szerepe eltérő.

### Training split

A preprocessing és a modellparaméterek illesztésére szolgál.

### Validation split

Használata:

- modellek összehasonlítása;
- hiperparaméter-döntések;
- early stopping;
- kísérleti döntések.

### Test split

Csak a végső generalization reporting során használjuk.

A test split nem vesz részt architektúra- vagy hiperparaméter-hangolásban.

## StandardScaler leakage protection

A scaler kizárólag a training adaton illeszkedik:

```text
fit(train)
transform(train)
transform(validation)
transform(test)
```

Ez alapvető data-leakage elleni döntés.

---

# 9. Training-data augmentation kísérlet

A hivatalos UCI datasetet soha nem írjuk felül, és nem állítjuk róla hamisan, hogy több valódi adatot tartalmaz.

Helyette külön **training-only SMOTE experiment** készült.

A logika:

```text
hivatalos dataset
→ train/validation/test split
→ StandardScaler fit az eredeti train adaton
→ train transzformáció
→ SMOTE kizárólag a transzformált train adaton
→ synthetic sorok inverse-transformja
→ train_augmented.csv külön mentése
```

Alapértelmezett konfiguráció:

```yaml
augmentation:
  enabled: true
  method: smote
  target_multiplier: 1.15
  k_neighbors: 5
```

A becsomagolt pre-executed preview esetén:

```text
eredeti training sorok : 9,527
augmentált training    : 19,985
synthetic sorok        : 10,458
```

Ez kifejezetten **synthetic interpolationként** van dokumentálva, nem újonnan gyűjtött valódi adatokként.

A validation és test adatok soha nem augmentálódnak.

A projekt augmentation ablation kísérletet is futtat:

```text
ugyanaz a modell + eredeti train
vs.
ugyanaz a modell + SMOTE-augmentált train
```

reprezentatív algoritmusokon.

---

# 10. STEP 04 — Heterogén modellbenchmark

A projektet valódi model-family benchmarkká bővítettük, hogy ne feltételezzük automatikusan, hogy a neurális háló a legjobb választás tabuláris adatokon.

A benchmark algoritmusai:

| Modellcsalád | Algoritmus |
|---|---|
| Triviális referencia | Dummy Most Frequent |
| Lineáris | Logistic Regression |
| Dimenziócsökkentés + lineáris | PCA + Logistic Regression |
| Generatív lineáris | Linear Discriminant Analysis |
| Valószínűségi | Gaussian Naive Bayes |
| Instance-based | K-Nearest Neighbors |
| Kernel | RBF Support Vector Machine |
| Egyetlen döntési fa | Decision Tree |
| Bagging | Random Forest |
| Randomizált fák | Extra Trees |
| Boosting | HistGradientBoosting |
| Boosting | XGBoost |
| Boosting | LightGBM |
| Boosting | CatBoost |
| Ensemble | Soft Voting |

A PyTorch MLP később ugyanezek mellett kerül értékelésre.

A cél a különböző **inductive biasok** összehasonlítása, nem annak feltételezése, hogy egyetlen modellcsalád biztosan nyer.

---

# 11. Cross-validation és model selection

A klasszikus benchmark kombinálja:

```text
training-only stratified cross-validation
+
külön validation holdout
```

A cross-validation méri:

```text
CV Macro F1 mean
CV Macro F1 standard deviation
```

Ez egyszerre ad képet:

```text
minőségről
+
stabilitásról
```

A test set érintetlen marad a STEP 06-ig.

---

# 12. Metrikák

A benchmark, ahol az algoritmus támogatja, az alábbiakat rögzíti:

- Accuracy;
- Balanced Accuracy;
- Macro Precision;
- Macro Recall;
- Macro F1;
- Weighted F1;
- Matthews Correlation Coefficient;
- Top-2 Accuracy;
- Multiclass Log Loss;
- One-vs-Rest Macro ROC-AUC;
- Cross-validation Macro F1 mean;
- Cross-validation Macro F1 standard deviation;
- Fit time;
- Inference latency per sample.

Így a model selection nemcsak prediktív teljesítményt, hanem működési szempontokat is figyelembe tud venni.

---

# 13. STEP 05 — PyTorch neurális háló

A neurális modell egy kompakt fully connected MLP.

Tipikus architektúra:

```text
16 input
→ 64 hidden unit
→ 32 hidden unit
→ 7 logit
```

Training stack:

```text
PyTorch Dataset / DataLoader
CrossEntropyLoss
Adam
ReLU
Dropout
early stopping
best-checkpoint restoration
```

Epochonként rögzítésre kerül:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy.

A háló szándékosan nem túlméretezett, mert a projekt fő célja a helyes supervised-learning workflow bemutatása tabuláris adatokon.

---

# 14. Hiperparaméter-benchmark

Több MLP konfiguráció kerül összehasonlításra.

Vizsgált paraméterek:

```text
hidden layer dimensions
dropout
learning rate
batch size
```

A vizualizáció olvasható konfiguráció-információkat és pontos Macro F1 értékeket mutat.

---

# 15. STEP 06 — Végső kiértékelés

Miután a fejlesztési döntések lezárultak, a candidate modellek az érintetlen test spliten kerülnek értékelésre.

A végső evaluation tartalma:

- all-model comparison;
- normalized confusion matrix;
- classonkénti precision / recall / F1;
- misclassified samples;
- prediction confidence;
- Top-2 behavior;
- model comparison visualization;
- permutation feature importance.

A workflow tehát külön kezeli:

```text
model selection
és
final reporting
```

folyamatát.

---

# 16. Error analysis

A hibás predikciók elemzése több dimenzióban történik:

```text
actual class
predicted class
maximum predicted probability
top-2 margin
```

Ez lehetővé teszi például:

- mely class párok keverednek gyakran;
- a hibás predikciók jellemzően bizonytalanok-e;
- vannak-e high-confidence hibák;
- mely esetek alkalmasak manual review-ra.

---

# 17. Prediction-confidence elemzés

A helyes és hibás predikciók csoportjai külön-külön százalékosan normalizáltak.

Az ábra megmutatja:

- sample countokat;
- helyes predikciók medián confidence-ét;
- hibás predikciók medián confidence-ét.

Fontos következtetés:

> A softmax confidence használható uncertainty signalnak, de nem automatikusan kalibrált helyességi valószínűség.

---

# 18. Selective classification / manual review

A projekt confidence-threshold policy-t is vizsgál:

```text
ha confidence >= threshold:
    automatikus elfogadás
különben:
    manual review
```

Minden thresholdnál mérjük:

```text
coverage
accepted-sample accuracy
manual-review count
selective error rate
```

A várható trade-off:

```text
threshold ↑
→ coverage ↓
→ accepted-sample accuracy ↑
```

---

# 19. Model-agnostic feature importance

A legerősebb klasszikus modellen permutation importance készül.

Módszer:

```text
egy feature permutálása
→ score romlásának mérése
→ nagyobb romlás = fontosabb prediktív feature
```

Ez model-agnostic értelmezési réteget ad.

---

# 20. STEP 07 — Inference pipeline

Példa:

```bash
python predict.py --input example.json
```

Inference flow:

```text
raw JSON
→ schema validation
→ mentett StandardScaler
→ mentett label encoder
→ mentett PyTorch modell
→ logits
→ softmax probabilities
→ predicted class + confidence
```

Az inference notebook foglalkozik még:

- hibás inputtal;
- hiányzó feature-rel;
- rossz adattípussal;
- mini-batch inference-szel;
- manual-review thresholdokkal.

---

# 21. STEP 08 — Monitoring és drift simulation

A projekt synthetic covariate driftet szimulál kiválasztott feature-ök eloszlásának eltolásával.

Monitoring metrikák:

```text
Population Stability Index (PSI)
Kolmogorov-Smirnov statistic
KS p-value
feature-level drift severity
```

A válaszstratégia:

```text
drift detected
→ data-quality check
→ slice analysis
→ labelled performance audit
→ retraining decision
```

---

# 22. Visualization refactor

A reusable plotting logika helye:

```text
src/dry_bean/visualization.py
```

Ugyanaz a plotting function használható notebookból és automatizált report generálásból is.

Fontosabb plotok:

```text
class distribution
feature distributions
correlation matrix
PCA
training loss
training accuracy
hyperparameter comparison
normalized confusion matrix
per-class F1
prediction confidence
model comparison
data drift
classical benchmark
CV stability
multi-metric comparison
performance vs latency
permutation importance
augmentation balance
augmentation ablation
selective classification
```

---

# 23. Notebook-filozófia

A notebookokat szándékosan **nem** redukáltuk egyszerű wrapper callokra.

Tartalmaznak:

- célkitűzést;
- elméleti hátteret;
- valódi kódhívásokat;
- köztes értékeket;
- táblázatokat;
- vizualizációkat;
- interpretációt;
- mérnöki döntéseket;
- következtetéseket;
- next-step útmutatást.

A reusable implementáció továbbra is a `src/` alatt található.

---

# 24. HU / EN notebook-dokumentáció

A notebook rendszer támogatja a nyelvváltást külön notebookkészletek fenntartása nélkül.

Használat:

```bash
python switch_language.py --language en
python switch_language.py --language hu
python switch_language.py --status
```

A váltás megőrzi:

- code cellákat;
- execution countokat;
- DataFrame outputokat;
- plotokat;
- embedded PNG-ket;
- model outputokat.

---

# 25. Tesztelési stratégia

A projekt célzott teszteket tartalmaz:

```text
03_tests/
```

Tesztelt területek:

- schema validation;
- config betöltés;
- preprocessing;
- augmentation;
- benchmark contractok;
- PyTorch model input/output;
- inference;
- selective classification;
- monitoring logic;
- runtime environment viselkedés.

---

# 26. Reprodukálhatóság

A konfiguráció központi helye:

```text
config.yaml
```

Tartalma többek között:

- random seed;
- adatútvonalak;
- split arányok;
- augmentation settings;
- benchmark settings;
- PyTorch training paraméterek;
- drift settings;
- result pathok;
- artifact pathok.

A reusable kódból eltávolítottuk az abszolút Windows pathokat.

---

# 27. Result és artifact struktúra

A generált outputok elkülönülnek az implementációtól:

```text
04_results/
├── figures/
├── metrics/
├── models/
└── predictions/
```

Példák:

```text
model_bundle.pt
best_classical_model.joblib
scaler.joblib
label_encoder.joblib

baseline_results.csv
model_comparison.csv
hyperparameter_results.csv
test_metrics.json
drift_report.csv

test_predictions.csv
misclassified_samples.csv
example_prediction.json
```

---

# 28. Pre-executed portfolio preview

A becsomagolt portfólió ZIP előre lefuttatott notebook outputokat és ábrákat tartalmaz, így egy reviewer azonnal meg tudja nézni a projektet.

A preview egy determinisztikus, schema-kompatibilis surrogate adaton készült, mert a build környezetből nem volt elérhető az UCI download endpoint.

Ezért:

```text
04_results/preexecuted_preview/
```

presentation/demo eredményeket tartalmaz.

Ezeket **nem szabad hivatalos UCI benchmarkként kommunikálni**.

A normál runtime pipeline továbbra is a hivatalos UCI adathalmazt használja.

---

# 29. Kétparancsos használat

Első setup:

```bash
python 00_setup_project.py
```

Teljes futtatás:

```bash
python run_project.py
```

Windows alatt, ha a globális `python` parancs nincs megfelelően konfigurálva:

```powershell
.\.venv\Scripts\python.exe run_project.py
```

---

# 30. Mit bizonyít a projekt?

A projekt fő üzenete nem az, hogy:

> „Betanítottam egy neurális hálót.”

Hanem az, hogy:

> „Képes vagyok megtervezni, validálni, összehasonlítani, kiértékelni, operationalizálni és monitorozni egy teljes multiclass Machine Learning workflow-t.”

A projekt demonstrálja:

```text
data contracts
reprodukálható environment
EDA
leakage prevention
class imbalance handling
synthetic augmentation experiment
model-family benchmarking
cross-validation
holdout model selection
PyTorch training
hyperparameter experimentation
multi-metric evaluation
error analysis
uncertainty-aware decision policy
inference contract
model-agnostic interpretation
drift detection
testing
reproducible reporting
```

---

# 31. Korlátok

### A SMOTE nem hoz létre valódi megfigyeléseket

A synthetic sorok interpolációk, és nem biztos, hogy fizikailag megfigyelt babmintáknak felelnek meg.

### A probability calibration nincs teljesen kidolgozva

Lehetséges további fejlesztések:

```text
temperature scaling
Platt scaling
isotonic calibration
Expected Calibration Error
reliability diagram
```

### A drift simulation synthetic

Valódi monitoringhoz production adatok és lehetőség szerint késleltetett ground-truth label-ek kellenének.

### Az adathalmaz scope-ja korlátozott

Production rendszerben vizsgálni kellene többek között az acquisition conditionöket, batch effecteket, domain shiftet, data lineage-et és valós class priorokat.

---

# 32. További fejlesztési lehetőségek

Erős következő lépések:

```text
probability calibration
SHAP analysis
Optuna / Bayesian tuning
repeated stratified CV
experiment tracking
MLflow
batch inference API
FastAPI model serving
Docker packaging
CI test workflow
model/version registry
real monitoring dashboard
```

---

# 33. Ajánlott reviewer útvonal

Egy recruiter vagy ML Engineer gyorsan megértheti a projektet ebben a sorrendben:

```text
1. README.md
2. workflow diagram
3. STEP 02 EDA notebook
4. STEP 03 preprocessing + SMOTE methodology
5. STEP 04 algorithm benchmark
6. STEP 05 PyTorch training
7. STEP 06 evaluation/error analysis
8. STEP 07 inference
9. STEP 08 monitoring
```

Részletesebb módszertan:

```text
reports/PROJECT_METHODOLOGY.md
reports/ALGORITHM_BENCHMARK.md
reports/DATA_AUGMENTATION_METHODOLOGY.md
reports/TECHNICAL_WALKTHROUGH.md
```

---

# 34. Végső összefoglalás

A végső repository egy hagyományos klasszifikációs feladatból egy strukturált Machine Learning engineering projektté fejlődött.

Az alapdesign:

```text
Notebook
= magyarázat és elemzés

src/
= reusable implementáció

workflow/*.py
= automatizált egyedi lépések

run_project.py
= end-to-end orchestration

04_results/
= generált artifactok

03_tests/
= regression protection
```

A projekt szándékosan egyensúlyoz az alábbiak között:

```text
egyszerűség
modularitás
reprodukálhatóság
oktatási érték
mérnöki minőség
portfólió-olvashatóság
```

anélkül, hogy egy közepes méretű ML projektből indokolatlanul összetett framework lenne.

---

## Rövid portfólió-összefoglaló

**Dry Bean Classification — Production-style Multiclass ML System**

End-to-end multiclass ML pipeline a UCI Dry Bean dataseten, strict schema validationnel, leakage-safe preprocessinggel, klasszikus model-family benchmarkkal, training-only SMOTE augmentationnel, PyTorch MLP tuninggal, multi-metric evaluationnel, error/confidence analysis-szel, selective classificationnel, JSON inference-szel, drift monitoringgal, reusable vizualizációkkal, automatikus setuppal, tesztekkel és egyparancsos pipeline futtatással.

Fő technológiák:

```text
Python
Pandas
scikit-learn
PyTorch
imbalanced-learn
XGBoost
LightGBM
CatBoost
Matplotlib
PyYAML
pytest
Jupyter
Graphviz
```
