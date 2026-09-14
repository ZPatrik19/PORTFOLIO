# Hibaelhárítás

## Python command not found
Telepíts Python 3.10–3.14 verziót, majd nyisd újra a terminált. Windowson a setup a `py` launchert is tudja használni.

## `ModuleNotFoundError: prompt_benchmark`
A repository rootból futtasd: `python -m pip install -e .`, vagy használd a mellékelt launchereket.

## Hiányzó API key
Mock/Ollama credential nélkül használható; cloud providerhez add meg a kulcsot az UI/runtime environmentben. Ne hardcode-old source fájlba.

## Provider authentication error
Ellenőrizd a kulcsot, provider választást, modellnevet, projekt/billing/free-tier státuszt és API-hozzáférést. Az autentikációs hibákat a rendszer szándékosan nem retry-olja.

## Quota / 429
Csökkentsd a benchmark méretét, várd meg a quota resetet, használj ingyenes/lokális providert, vagy támaszkodj a resumable history/checkpoint funkcióra. Retry csak átmeneti rate-limit hibánál indokolt.

## Playground chart hiba
A legfrissebb verziót használd: a token reshape-ra regressziós teszt védi a korábbi pandas `melt` oszlopnévütközést.

## A 8501 port foglalt
Állítsd le a korábbi Streamlit processzt, vagy indítsd más porton: `prompt-benchmark ui --port <másik-port>`.

## Docker build sikertelen
Ellenőrizd a Docker elérhetőségét, package index hálózatot, és hogy `.venv`/nagy output ne kerüljön build contextbe. A `.dockerignore` ezt kezeli.

## Kubernetes CrashLoopBackOff
Nézd meg a `kubectl logs` és `kubectl describe pod` kimenetet, environment/secret beállítást, memória limitet és Streamlit startup hibákat.
