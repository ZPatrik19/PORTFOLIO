# Adatpipeline

## Források
- `mock`: lokális szintetikus Challenge Set determinisztikus offline validációhoz.
- `huggingface`: opcionális külső support-ticket dataset generalization checkhez.
- `sample`: kisméretű lokális fejlesztői adathalmaz.
- UI CSV upload: felhasználó által feltöltött címkézett evaluation set.

## Kanonikus séma
A `sample_id` egyedi és nem üres, a `text` nem üres, a `true_label` pedig a hat támogatott címke egyikéhez tartozik. Challenge adatokhoz `case_type`, `difficulty`, `scenario_id` és megjegyzések is tartozhatnak.

## Split policy
Alapértelmezett szintetikus forrás: 10 800 sor. Final holdout: 6 000. Development: 3 000. Few-shot könyvtár: 24. A fix random seed a `configs/benchmark.yaml` fájlban található.

## Leakage-védelem
A development, holdout és few-shot kiválasztások diszjunktak; a fine-tuning export csak development adatot használ; duplikált ID és ismeretlen label inference előtt hibázik; a cache elutasításra kerül, ha a sample identitása megváltozik.

## Pilot mintavétel
A kis futások determinisztikus, label- és scenario-reprezentatív mintavételt használnak `head(n)` helyett, így kisebb az esélye a félrevezető, csak könnyű példákból álló pilotnak.
