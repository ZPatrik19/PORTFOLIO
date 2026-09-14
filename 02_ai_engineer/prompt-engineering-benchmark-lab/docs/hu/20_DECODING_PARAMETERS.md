# Decoding paraméterek

A prompt tartalma és a decoding beállítás külön független változó. Ezért a parameter lab fix prompt mellett egy változót módosít egyszerre.

## Temperature
A véletlenszerűséget szabályozza. Klasszifikációnál általában alacsony érték indokolt a stabilitás miatt. Példa sweep: `0.0, 0.2, 0.5, 0.8`.

## Top-p
Nucleus sampling küszöb. Példa sweep: `0.5, 0.8, 0.95, 1.0`.

## Top-k
A K legvalószínűbb tokenre korlátozza a jelölteket. Példa sweep: `10, 20, 40, 80`. Nem minden provider támogatja.

## Capability-aware működés
A provider capability metadata megakadályozza nem támogatott mezők elküldését. Az egységes UI nem jelenti azt, hogy minden API ugyanazt tudja.

## Értelmezés
Determinisztikus klasszifikációnál a stochastic beállítások miatti quality/reliability romlás gyakran fontosabb, mint a kreatív diverzitás. Szabad szöveggenerálásnál más lehet a trade-off.
