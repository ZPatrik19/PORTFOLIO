# Advanced Prompt Engineering – részletes útmutató

Ez a projekt nem promptgyűjtemény. Minden technika ugyanazon klasszifikációs feladaton, ugyanazon holdout adaton és azonos provider/model mellett mérhető.

## 1. Prompt anatomy: mit jelent a Persona / Instruction / Context / Audience / Tone / Data / Format?

A P16 `Full Advanced Template` ezeket külön részekre bontja:

- **Persona** – milyen szerepben dolgozik a modell. Példa: senior SaaS support-routing classifier.
- **Instruction** – pontosan mit kell végrehajtani. Példa: válassz pontosan egy routing labelt.
- **Context** – miért történik a feladat, milyen üzleti környezetben használjuk a választ.
- **Audience** – ki fogyasztja a választ. Itt downstream program, nem ember.
- **Tone** – milyen kommunikációs stílus szükséges. Klasszifikációnál terse/deterministic.
- **Reference data** – label-definíciók, szabályok, üzleti tudás.
- **Examples** – few-shot vagy contrastive few-shot példák.
- **Constraints** – mit szabad és mit nem szabad tenni.
- **Input data** – a customer ticket, erős delimiterrel elkülönítve.
- **Format** – plain label, JSON vagy provider által enforced JSON Schema.

A lényeg: nem feltételezzük, hogy mindegyik komponens javít. A benchmark megméri a hatást és a token overheadet.

## 2. P0–P16 kísérletek

### P0 – Zero-shot baseline
A minimális kontrollcsoport. E nélkül nem tudjuk, hogy az összetettebb prompt valóban hozzáadott értéket ad-e.

### P1 – Label definitions
Explicit szemantika az osztályokhoz. Elsősorban a hasonló intentek – például API vs technical – elkülönítését célozza.

### P2 – Role/System prompt
A feladat identitását magasabb prioritású system instructionként adja meg.

### P3 – Few-shot
Train/development oldalról származó demonstrációk. A final benchmark minták SOHA nem lehetnek example-ok.

### P4 – Explicit constraints
Pontosan egy label, nincs új kategória, nincs extra magyarázat. Fő cél: output reliability.

### P5 – Decision policy
Determinista precedence szabályok a több intentet tartalmazó vagy határeset ticketekhez.

### P6 – Prompt-only JSON
A prompt kéri a JSON-t. Itt még nincs API-szintű kényszerítés, ezért mérhető az `invalid_json_rate`.

### P7 – Structured Output
JSON Schema / structured output API feature. Ezt külön kezeljük, mert már nem tisztán prompt engineering, hanem constrained generation is.

### P8 – Persona
Erősebb domain persona. A cél annak mérése, hogy önmagában a persona jelent-e minőségi javulást.

### P9 – Instruction + Context blocks
Explicit blokkok: `[INSTRUCTION]`, `[CONTEXT]`, `[REFERENCE DATA]`, `[CONSTRAINTS]`, `[INPUT DATA]`, `[OUTPUT]`.

### P10 – Format + Audience + Tone
Megadja, hogy downstream gép a fogyasztó, ezért nincs conversational prose, csak stabil formátum.

### P11 – Delimited Data
A customer ticket untrusted data. `<ticket>...</ticket>` delimiterek segítenek szétválasztani a felhasználói adatot és a rendszerutasítást.

### P12 – Contrastive Few-shot
Nem csak jó példákat ad, hanem határeseteket is: miért cancellation és nem billing, miért API és nem technical.

### P13 – Reasoning-model mode
Ha a provider támogat reasoning effort / thinking módot, azt használhatja. A projekt NEM kér hidden chain-of-thought kiírást. Csak a végső labelt értékeljük.

### P14 – Tree-of-Thought-inspired Branch + Vote
Három független szakértői ág:
1. lexical intent specialist,
2. policy specialist,
3. ambiguity specialist.

Mindhárom csak végső labelt ad. A rendszer deterministic majority vote-tal aggregál. Ez mérhetővé teszi a multi-branch megközelítést anélkül, hogy private chain-of-thoughtot gyűjtenénk.

### P15 – Grammar / Schema Constrained Output
Ahol a provider támogatja, JSON Schema / constrained output kényszeríti a grammar-szerű szerkezeti követelményt. Mérjük a `json_grammar_valid_rate` értéket.

### P16 – Full Advanced Template
A teljes prompt architecture: persona + instruction + context + audience + tone + reference data + examples + constraints + delimiters + format.

Fontos hipotézis: **a leghosszabb prompt nem feltétlenül a legjobb prompt**. Ezért F1 mellett token, latency és cost is mérve van.

## 3. Mit mérünk?

- Accuracy
- Macro Precision / Recall / F1
- Weighted F1
- Per-class Precision / Recall / F1
- Confusion matrix
- Invalid output rate
- Invalid JSON rate
- Output-contract validity
- JSON grammar validity
- Input/output/total tokens
- Mean / median / P95 latency
- Estimated cost/request és cost/1000 request
- Bootstrap 95% confidence interval
- Fixed vs regressed samples
- Hard examples

## 4. Miért Macro F1?

A Macro F1 minden osztálynak azonos súlyt ad. Ez támogatja azt a döntést, hogy ne legyen egy prompt látszólag jó csak azért, mert a könnyebb vagy gyakoribb kategóriákon jól teljesít.

## 5. Mit NE csináljunk?

- ne fejlesszük a promptot folyamatosan a final holdout alapján;
- ne nevezzük a mock eredményt LLM benchmarknak;
- ne hasonlítsunk különböző modellt úgy, mintha csak a prompt változott volna;
- ne keverjük egy grafikonba a prompt-stratégia és a sampling-parameter hatását kontroll nélkül;
- ne kérjük vagy tároljuk a modell hidden chain-of-thoughtját.
