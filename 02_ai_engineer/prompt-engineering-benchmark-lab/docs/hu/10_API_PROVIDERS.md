# API providerek

## Támogatott providerek
- **Mock:** determinisztikus offline promptérzékeny szimulátor; nincs kulcs és nincs valódi modellminőség-állítás.
- **Ollama:** lokális modell runtime; nincs cloud key, latency hardverfüggő.
- **Gemini:** Google GenAI SDK-n keresztüli cloud provider.
- **Groq:** cloud inference provider saját capability/rate-limit profillal.
- **OpenRouter:** többmodell-routoló provider.
- **OpenAI:** cloud provider, modelltől/API képességtől függő structured-output támogatással.

## Közös adapter contract
Minden provider ugyanarra a response objektumra normalizálódik: raw output, parsed label, input/output token, latency, validity flag, error, model/provider metadata és opcionális cost.

## Reliability policy
- explicit timeout;
- átmeneti provider hibánál korlátozott exponential backoff;
- authentication/invalid-request fail-fast;
- malformed response strukturált hibaként jelenik meg, nincs lenyelve;
- benchmark checkpoint miatt quota/rate-limit megszakítás után folytatható a futás.

## Secretek
A kulcsok runtime secretek. A `.env` gitignore-olt, az `.env.example` csak placeholdereket tartalmaz. Az UI az aktuális processhez fogadhat kulcsot, de az nem kerülhet source-ba, result CSV-be, logba, Dockerfile-ba vagy Kubernetes példa manifestbe.
