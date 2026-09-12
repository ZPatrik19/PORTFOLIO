# Smart Battery Energy Management — Reinforcement Learning Decision System

## 1. Projektcél és problémafelvetés

A projekt célja egy reinforcement learning alapú akkumulátor-energiagazdálkodási rendszer létrehozása, amely időben egymást követő döntéseket hoz egy energiatároló rendszer töltéséről és kisütéséről.

A központi kérdés:

> Hogyan tanulhat meg egy agent olyan akkumulátor-töltési és kisütési stratégiát, amely változó energiaár, fogyasztás és megújuló termelés mellett hosszú távon kedvező működést biztosít?

A probléma azért érdekes, mert a jelenlegi döntés közvetlenül megváltoztatja a rendszer jövőbeli állapotát.

Például egy magas energiaár mellett végrehajtott kisütés rövid távon csökkentheti a költséget, ugyanakkor csökkenti az akkumulátor State of Charge értékét, így később kevesebb energia áll rendelkezésre.

Ezért a probléma nem egyszerű predikciós feladat, hanem:

```text
Sequential Decision-Making Problem
```

ahol a hosszú távú következményeket is figyelembe kell venni.

A projekt négy területet kapcsol össze:

```text
Electrical Engineering
+
Energy Systems
+
Optimization
+
Reinforcement Learning
```

---

## 2. Üzleti és mérnöki cél

A valódi cél nem egyszerűen a cumulative reward maximalizálása.

A rendszernek több, egymással részben konfliktusban álló célt kell kezelnie:

* villamosenergia-költség csökkentése;
* hálózati csúcsterhelés mérséklése;
* megújuló energia jobb helyi felhasználása;
* akkumulátor felesleges ciklizálásának mérséklése;
* fizikai korlátok betartása.

A projekt ezért az RL reward mellett külön energetikai és üzleti KPI-ok alapján is értékeli a policy-kat.

Ez azért fontos, mert egy magas cumulative reward önmagában még nem bizonyítja, hogy a controller valóban jó energetikai megoldás.

---

## 3. Miért Reinforcement Learning?

A probléma első fontos kérdése az volt, hogy valóban szükséges-e reinforcement learning.

Supervised learning esetén tipikusan egy:

```text
X → y
```

leképezést tanulnánk.

Ebben a problémában azonban nincs minden állapothoz előre megadott „helyes” battery action.

Ráadásul:

```text
Current state
    ↓
Action
    ↓
Battery SOC changes
    ↓
Future state changes
    ↓
Future available actions change
    ↓
Future cost changes
```

A jelenlegi action tehát befolyásolja a jövőbeli lehetőségeket.

Ez teszi a problémát természetes reinforcement learning feladattá.

---

## 4. Lehetséges megoldási stratégiák

A reinforcement learning nem az egyetlen lehetséges megközelítés.

A projektben ezért több alternatívát is figyelembe vettem.

### No Battery

Az akkumulátor soha nem kerül használatra.

Ez szolgál alapreferenciaként az energiaköltséghez és a peak demandhez.

### Random Policy

Véletlenszerű, de érvényes actionöket választ.

Ennek célja elsősorban sanity check: egy tanult policy-nak egyértelműen jobbnak kell lennie a véletlenszerű vezérlésnél.

### Rule-Based Controller

Egyszerű szakértői szabályok alapján működik:

```python
if price < low_threshold:
    charge
elif price > high_threshold:
    discharge
else:
    idle
```

Ez különösen fontos baseline, mert egyszerű, olcsó és jól értelmezhető.

Ha egy komplex RL agent nem képes ezt meghaladni, akkor alkalmazása üzletileg nem feltétlenül indokolt.

### Mathematical Optimization

További alternatíva lehet:

```text
Linear Programming
Mixed Integer Programming
Model Predictive Control
Dynamic Programming
```

Ha például a teljes következő napi energiaár ismert, egy optimalizáló nagyon erős referencia-megoldást adhat.

A projektben ezért az RL-t nem automatikusan „jobb” módszerként, hanem egy lehetséges megoldási stratégiaként kezelem.

---

## 5. A reinforcement learning environment

A problémát saját Gymnasium-kompatibilis environmentként modelleztem.

Egy időlépés:

```text
1 hour
```

egy episode pedig tipikusan:

```text
24 hours
```

vagy hosszabb időszak.

Az RL ciklus:

```text
Environment
     │
     │ observation
     ▼
   Agent
     │
     │ action
     ▼
Battery Controller
     │
     │ executed action
     ▼
Energy System
     │
     │ reward + next state
     └──────────────────► Agent
```

A saját environment elkészítése fontos része volt a projektnek, mert így az olyan alapfogalmak, mint:

```text
State
Action
Reward
Transition
Episode
Policy
```

nem egy előre elkészített környezet mögött rejtőznek.

---

## 6. State Space

Az agent minimális observation vectorja:

```python
state = [
    battery_state_of_charge,
    electricity_price,
    electricity_demand,
    renewable_generation,
    hour_of_day,
]
```

### Battery State of Charge

Megmutatja, mennyi energia áll rendelkezésre az akkumulátorban.

Ugyanaz a magas energiaár teljesen más döntést eredményezhet 20% és 90% SOC mellett.

### Electricity Price

A battery arbitrage egyik legfontosabb információja.

A controller várhatóan olcsóbb időszakokban tölt, magasabb ár mellett pedig energiát használ fel az akkumulátorból.

### Electricity Demand

A rendszer aktuális fogyasztása.

Nagy demand mellett a battery discharge a peak grid demand csökkentésére is használható.

### Renewable Generation

A helyben termelt energia.

Nagy renewable generation esetén előnyös lehet az energia akkumulátorban történő eltárolása.

### Hour of Day

Az ár, fogyasztás és renewable generation jellemzően időbeli struktúrával rendelkezik.

Az időinformáció segít az agentnek ezeket a napi mintázatokat megtanulni.

---

## 7. Future leakage elkerülése

Fontos tervezési szabály volt, hogy az agent csak olyan információhoz férjen hozzá, amely a döntés pillanatában valóban rendelkezésre állhat.

Nem megengedett például:

```text
actual_future_price
actual_future_demand
```

közvetlen használata.

Használható viszont:

```text
forecasted_price
forecasted_demand
forecasted_generation
```

ha ezek valódi forecastként állnak rendelkezésre.

A:

```text
future actual value
```

és a:

```text
forecast available at decision time
```

nem ugyanaz.

Ez különösen fontos időfüggő ML és RL rendszereknél.

---

## 8. Action Space és akkumulátorfizika

Az első environment diszkrét action space-t használ:

```text
0 = charge
1 = idle
2 = discharge
```

Ez jól használható:

```text
Q-Learning
DQN
```

algoritmusokkal.

A fejlettebb változat continuous action space-t használ:

```text
[-1, 1]
```

ahol:

```text
-1 = maximum discharge
 0 = idle
+1 = maximum charge
```

Ez közelebb áll egy valódi invertervezérléshez.

Az environment modellezi legalább:

```text
battery_capacity
state_of_charge
maximum_charge_rate
maximum_discharge_rate
charging_efficiency
discharging_efficiency
minimum_soc
maximum_soc
```

értékeket.

---

## 9. Requested Action vs Executed Action

Az agent nem írhatja felül a fizikai korlátokat.

Ha például az akkumulátor:

```text
SOC = 98%
```

és az agent maximális töltést kér, nem lehet ugyanakkora teljesítményt végrehajtani, mint 40% SOC mellett.

Ezért külön tároljuk:

```text
requested_action
```

és:

```text
executed_action
```

értékeket.

A folyamat:

```text
Agent action
    ↓
Battery constraints
    ↓
Action clipping
    ↓
Executed action
```

Ez közelebb áll egy production vezérlőrendszerhez, ahol egy safety/physics layer mindig felülírhatja az intelligens controller fizikailag lehetetlen kérését.

---

## 10. Energy Balance

Az energiaegyensúly alapja:

```text
grid_energy =
    demand
    - renewable_generation
    + battery_charging
    - battery_discharging
```

A charging és discharging efficiency külön kerül figyelembevételre.

Ez azért szükséges, mert az akkumulátor nem veszteségmentesen működik.

A modellezés során ezért külön kell kezelni:

```text
energy entering/leaving the battery
```

és:

```text
energy seen by the grid/load
```

értékeket.

---

## 11. Reward Function

A reward tervezése az RL projekt egyik legfontosabb része.

Egy egyszerű megoldás lenne:

```python
reward = -electricity_cost
```

Ez azonban nem reprezentálja az összes valódi rendszerkövetelményt.

Ezért a reward több komponensből áll:

```text
reward =
    - electricity_cost
    - peak_demand_penalty
    - battery_degradation_penalty
    - constraint_violation_penalty
```

### Electricity Cost

A fő gazdasági objective.

### Peak Demand Penalty

A nagy hálózati teljesítményigény visszaszorítására szolgál.

### Battery Degradation Penalty

Megakadályozza, hogy az agent gyakorlatilag ingyenes erőforrásként kezelje az akkumulátor ciklizálását.

### Constraint Violation Penalty

Arra ösztönzi az agentet, hogy megtanulja a rendszer fizikai határait.

---

## 12. Reward Shaping

Az RL egyik fontos problémája, hogy:

> Az agent azt optimalizálja, amit rewardként definiálunk, nem feltétlenül azt, amit eredetileg szerettünk volna.

Például túl nagy degradation penalty mellett az agent számára optimális megoldás lehet:

```text
never use the battery
```

Ha viszont nincs degradation penalty, az agent túlzottan gyakran tölthet és süthet ki.

Ezért a projektben reward ablation study is található:

```text
without degradation penalty
vs
with degradation penalty
```

A cél annak megértése, hogyan változtatja meg a reward design a learned policy viselkedését.

---

## 13. Adatok és adatfeltárás

A minimálisan szükséges idősoros adatok:

```text
timestamp
electricity_price
electricity_demand
renewable_generation
```

A projekt első verziója kontrollált, reprodukálható szintetikus adatot használ.

Ennek előnye:

* nincs licencprobléma;
* minden futtatás reprodukálható;
* egyszerűen készíthetők extrém stress scenario-k;
* az RL environment viselkedése könnyebben ellenőrizhető.

A későbbi verzióban ugyanaz a pipeline valódi energia-, smart-meter-, PV- vagy market price adatokra cserélhető.

Az EDA során elsősorban azt vizsgáljuk:

* milyen az energiaár napi profilja;
* hol vannak demand peak-ek;
* mikor magas a renewable generation;
* milyen az egyes változók variabilitása;
* vannak-e olyan időbeli mintázatok, amelyek alapján érdemes döntést hozni.

---

## 14. Train, Validation és Test

Idősoros probléma miatt nem használunk véletlenszerű train-test shuffle-t.

A helyesebb struktúra:

```text
Training Period
      ↓
Validation Period
      ↓
Test Period
```

A training időszak szolgál az agent tanítására.

A validation időszak használható:

* hyperparameter kiválasztásra;
* reward design összehasonlításra;
* state representation kiválasztásra.

A test időszak csak a végső generalizációs értékelésre szolgál.

Ez csökkenti a future leakage és data snooping veszélyét.

---

## 15. Algoritmusválasztás

A projektben az algoritmusokat fokozatosan építettem fel.

### Tabular Q-Learning

Első algoritmusként azért hasznos, mert közvetlenül megmutatja:

```text
Q(s,a)
learning rate
discount factor
exploration
exploitation
Bellman update
```

fogalmait.

Hátránya, hogy a continuous state space-t diszkretizálni kell.

A state dimenzió növekedésével gyorsan jelentkezik a:

```text
curse of dimensionality
```

probléma.

### Deep Q-Network — DQN

A DQN a Q-table helyett neural networköt használ:

```text
state
  ↓
Q-Network
  ↓
Q(charge)
Q(idle)
Q(discharge)
```

Ez már képes continuous observationökből általánosítani.

Fontos komponensei:

```text
Experience Replay
Target Network
Bellman Target
Epsilon-Greedy Exploration
```

Diszkrét action space esetén természetes következő lépés a Q-Learning után.

### PPO

Continuous battery controlhoz a PPO alkalmasabb.

Itt az agent közvetlenül policy-t tanul:

```text
state
  ↓
Actor
  ↓
continuous battery action
```

A Critic közben:

```text
V(s)
```

értéket becsül.

A PPO ezért jól demonstrálja a:

```text
Actor-Critic
Policy Gradient
Continuous Control
```

megközelítést.

---

## 16. Milyen további algoritmusok jöhetnének szóba?

A projekt következő szintjén érdemes lenne összehasonlítani:

```text
SAC
TD3
Model Predictive Control
Linear / Mixed Integer Optimization
```

módszereket is.

Különösen érdekes összehasonlítás lenne:

```text
Rule-Based
vs
MPC
vs
MILP
vs
PPO
vs
SAC
```

ugyanazon battery environmenten.

Ez már közvetlenül hasonlítaná össze a klasszikus optimization és reinforcement learning megközelítéseket.

---

## 17. Értékelési stratégia

Az agentet nem csak cumulative reward alapján értékeljük.

A fő KPI-ok:

```text
Total Electricity Cost
Cost Saving %
Peak Grid Demand
Peak Reduction %
Renewable Self-Consumption
Battery Throughput
Battery Cycles
Constraint Violations
Average SOC
Episode Return
```

A legfontosabb üzleti összehasonlítás:

```text
Policy
vs
No Battery
vs
Rule-Based
```

Ha az RL policy csak rewardban jobb, de például lényegesen több battery cycle-t használ minimális cost saving mellett, akkor az eredmény nem feltétlenül előnyös.

---

## 18. Multiple Seeds

A reinforcement learning eredménye stochastic.

Egyetlen training run ezért nem elegendő.

A projekt több seedet használ:

```text
42
123
2026
```

és az eredményeket:

```text
mean ± standard deviation
```

formában értékeli.

Ez segít elkülöníteni:

```text
stable learning
```

és:

```text
one lucky run
```

eseteket.

---

## 19. Stress Test

A normál test időszak mellett több megváltozott scenario-t is használunk:

```text
Normal Prices
High Prices
Volatile Prices
High Renewable
Low Renewable
```

A cél annak vizsgálata, hogyan viselkedik a policy olyan környezetben, amely eltér a training distributiontől.

Ez egy egyszerű robustness és distribution-shift vizsgálat.

---

## 20. Ablation Study

Az ablation study célja annak megállapítása, hogy egy adott rendszerkomponens ténylegesen hozzáad-e értéket.

Két fontos kísérlet:

```text
Reward without degradation penalty
vs
Reward with degradation penalty
```

és:

```text
State without renewable forecast
vs
State with renewable forecast
```

Itt nem csak a végső rewardot, hanem az energiaköltséget, battery throughputot, cycle countot és más KPI-kat is összehasonlítjuk.

---

## 21. Policy Interpretability

A learned policy-t nem csak numerikus score-ként vizsgáljuk.

Különösen hasznos a:

```text
Electricity Price × Battery SOC
              ↓
         Selected Action
```

heatmap.

Egy logikus policy esetén például:

```text
Low price + low SOC
→ charge

Medium price
→ idle

High price + high SOC
→ discharge
```

mintázat várható.

Ha a learned policy teljesen kaotikus, az jelezhet:

* rossz rewardot;
* elégtelen traininget;
* hibás state representationt;
* nem megfelelő hyperparamétereket.

---

## 22. Projektstruktúra

A repository struktúrája tudatosan választja szét a tanulási workflow-t és a reusable implementációt.

```text
battery-energy-rl/
│
├── 00_setup_project.py
│
├── 01_data/
│   ├── raw/
│   └── processed/
│
├── workflow/
│   ├── step01_problem_and_data.ipynb
│   ├── step01_problem_and_data.py
│   ├── ...
│   ├── step09_policy_analysis.ipynb
│   └── step09_policy_analysis.py
│
├── src/
│   └── battery_rl/
│       ├── environment/
│       ├── agents/
│       ├── rewards.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── config.py
│
├── 03_tests/
│
├── 04_results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
│
├── 05_graphviz/
├── tools/
├── reports/
│
├── run_project.py
├── config.yaml
├── requirements.txt
└── README.md
```

A `workflow/` mappa mutatja a projekt logikai sorrendjét.

A notebookok célja:

```text
explanation
experimentation
visualization
learning
```

A `.py` fájlok célja:

```text
automation
reproducibility
pipeline execution
```

A reusable implementáció pedig a:

```text
src/
```

alatt található.

Így elkerülhető, hogy ugyanazt az algoritmust több notebookban külön-külön implementáljuk.

---

## 23. Reprodukálhatóság

A projekt első lépése:

```bash
python 00_setup_project.py
```

amely létrehozza a környezetet, telepíti a szükséges dependency-ket, előkészíti az adatokat és létrehozza a Jupyter kernelt.

A teljes workflow ezután egyetlen paranccsal is futtatható:

```bash
python run_project.py
```

Ez azért fontos, mert egy GitHub-portfólióprojektnek nem csak működnie kell a szerző gépén, hanem más számára is reprodukálhatónak kell lennie.

---

## 24. Tesztelés

Az RL eredmények csak akkor értelmezhetők, ha az environment fizikailag helyes.

Ezért külön teszteket kapnak például:

```text
SOC boundaries
charge/discharge limits
charging efficiency
discharging efficiency
energy balance
episode termination
requested vs executed action
```

Ha például az SOC fizikailag lehetetlen értéket vehetne fel, akkor az agent performance mérésének sem lenne jelentése.

---

## 25. Projektkorlátok

A projekt oktatási és portfólió célú, ezért több egyszerűsítést használ.

Jelenlegi korlátok például:

* szintetikus elsődleges dataset;
* egyszerűsített battery degradation modell;
* konstans charging/discharging efficiency;
* nincs részletes battery thermal model;
* nincs teljes hálózati power-flow modell;
* korlátozott forecast uncertainty;
* egyszerűsített electricity market modell.

Ezek tudatos modellezési döntések.

A lényeg, hogy dokumentálva legyenek, és világos legyen, hogyan lehetne a rendszert továbbfejleszteni.

---

## 26. Lehetséges továbbfejlesztések

A projekt következő verziója tartalmazhatna:

```text
real electricity market data
real PV / smart-meter data
probabilistic forecasting
temperature-dependent battery model
advanced degradation model
Model Predictive Control
MILP optimization
SAC / TD3
dynamic electricity tariffs
grid constraints
carbon-aware optimization
CityLearn benchmark
```

A saját egyszerű environment után a CityLearn logikus következő lépés:

```text
Custom Battery Environment
        ↓
RL fundamentals
        ↓
Validated policies
        ↓
CityLearn
        ↓
More realistic benchmark
```

---

## 27. Fő mérnöki döntések

A projekt során követett gondolkodás röviden:

```text
1. Először a valós problémát definiáltam.

2. Megvizsgáltam, hogy valóban sequential decision-making problémáról van-e szó.

3. Meghatároztam az üzleti és energetikai célokat.

4. Saját, fizikailag értelmezhető environmentet építettem.

5. Meghatároztam a state és action space-t.

6. Külön kezeltem a requested és executed actiont.

7. A rewardot több üzleti és fizikai komponensből építettem fel.

8. Egyszerű baseline-okkal kezdtem az értékelést.

9. Q-Learninggel építettem fel az RL alapokat.

10. DQN-nel neural network alapú Q-function approximationt használtam.

11. PPO-val continuous control irányba léptem tovább.

12. Train, validation és test időszakokat különítettem el.

13. Több random seed alapján értékeltem a stabilitást.

14. Stress testeket és ablation study-kat végeztem.

15. A policy működését vizualizációkkal is elemeztem.

16. A teljes workflow-t reprodukálható projektstruktúrába szerveztem.
```

---

## 28. Összegzés

A projekt fő célja nem egy reinforcement learning library használatának bemutatása.

A cél egy teljes mérnöki döntési rendszer megtervezése volt:

```text
Problem Definition
        ↓
Data
        ↓
Environment Design
        ↓
State / Action Design
        ↓
Reward Design
        ↓
Baseline Controllers
        ↓
Q-Learning
        ↓
DQN
        ↓
PPO
        ↓
Evaluation
        ↓
Stress Testing
        ↓
Ablation
        ↓
Policy Interpretation
```

A projekt legfontosabb üzenete:

> Nem egy előre elkészített reinforcement learning tutorialt futtattam le. Egy valós energiamenedzsment problémát sequential decision-making feladatként formalizáltam, fizikailag értelmezhető akkumulátor-környezetet készítettem, baseline kontrollereket és több reinforcement learning algoritmust hasonlítottam össze, majd a megoldásokat nemcsak reward, hanem üzleti és energetikai KPI-ok alapján is értékeltem.

Ez mutatja meg igazán a projektben az Electrical Engineering, Machine Learning és Reinforcement Learning szemlélet összekapcsolását.
