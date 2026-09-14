# Biztonság és reprodukálhatóság

## Secret kezelés
- A `.env` Gitből kizárt.
- Az `.env.example` csak placeholdereket tartalmaz.
- Provider kulcs nem kerül prompt template-be, result CSV-be, Docker image-be vagy Kubernetes példa YAML-ba.
- Authentication error nem retry-olódik.

## Reprodukálhatósági kontrollok
- fix dataset/split seed;
- determinisztikus reprezentatív pilot mintavétel;
- development/holdout/few-shot szeparáció;
- provider/model/settings/dataset identitást tartalmazó run manifest;
- raw prediction checkpoint;
- cache invalidáció adat- vagy sampling beállítás változásakor;
- pricing változáskor cost újraszámítás provider újrahívása nélkül.

## Nondeterminisztikus határ
Egy valós LLM provider lokális seed mellett is lehet nondeterminisztikus. A reprodukálhatóság ezért az exact input, prompt verzió, provider/model azonosító, sampling paraméterek, response telemetria és dataset snapshot megőrzését jelenti, nem bitazonos cloud válasz ígéretét.
