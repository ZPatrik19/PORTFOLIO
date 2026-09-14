# Mock szimuláció és metrikák

## Cél
A mock provider determinisztikus promptérzékeny szimulátor, amellyel kulcs és modell runtime nélkül végigjárható a teljes benchmark/UI/reporting stack.

## Mit szimulál?
A prompt feature-ök determinisztikusan módosítják a siker valószínűségét és a szimulált token/latency overheadet. Így advanced prompt lehet jobb, de drágább. Ez tudatos software-demo viselkedés, nem modell evidence.

## Explicit telemetria címkék
Mock soroknál `token_source = estimated_mock` és `latency_source = simulated_mock`, így a szimulált érték nem keverhető provider-reported usage-dzsal.

## Mi lesz valódi providernél?
A predikció, ahol elérhető a provider-reported usage, wall-clock request latency, SDK/API error és valós parser validity a tényleges futásból származik.

## Miért marad a szimulátor?
Determinisztikus regressziós lefedettséget ad dashboardra, historyra, metrikákra, error analysisre, cache-re és prompt workflow-ra quota/cost/network függés nélkül.
