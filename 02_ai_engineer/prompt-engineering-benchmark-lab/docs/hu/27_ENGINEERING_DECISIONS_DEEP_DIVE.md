# Engineering döntések — részletesen

1. **Klasszifikációs feladat:** objektív labelek mellett mérhető a promptkülönbség.
2. **Development + holdout:** csökkenti a prompt-overfitting kockázatát.
3. **Balanced core benchmark:** a domináns osztály nem rejti el a hibákat.
4. **Macro F1 elsődleges:** minden class azonos súlyú.
5. **Accuracy nem elég:** class-wise metrika és confusion matrix szükséges.
6. **Output validity külön:** productionben a parse-olható machine contract fontos.
7. **P6 vs P7:** prompt-only JSON és constrained generation eltérő mechanizmus.
8. **P95/P99 latency:** a tail latency fontos user/SLA szempontból.
9. **Capability matrix:** UI-szimmetria kedvéért nem küldünk unsupported paramétert.
10. **Nincs hidden CoT storage:** reasoninget végső döntéssel és külső branch aggregációval mérünk.
11. **Raw predikció mentés:** aggregate metrika kevés debughoz.
12. **Resumable benchmark:** API hiba/quota miatt ne kelljen újrafizetni a kész requesteket.
13. **Dataset-aware cache:** a sample ID önmagában nem elég identitás.
14. **Custom prompt:** a platform user hipotézist is mér, nem csak beépített recepteket.
15. **Fine-tuning export, nem auto-training:** elkerüli a váratlan költséget és provider lock-int.
16. **Multi-objective production döntés:** quality, reliability, token, latency, cost és complexity együtt számít.
