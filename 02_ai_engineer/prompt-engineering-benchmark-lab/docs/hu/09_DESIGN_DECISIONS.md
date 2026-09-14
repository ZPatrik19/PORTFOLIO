# Tervezési döntések

## Telepíthető package `sys.path` hack helyett
**Döntés:** a core logika a `03_src/prompt_benchmark` package-ben él és `pip install -e .` telepíti.  
**Indok:** ugyanaz az import viselkedés kell CLI-ből, notebookból, tesztből, Dockerből és Kubernetesből.

## Repository-relative pathok
**Döntés:** `ProjectPaths` + `pathlib`.  
**Indok:** megszünteti a fejlesztőgép-függést, és egységes Windows/Linux/container futást ad.

## Prompt benchmark és decoding sweep szétválasztása
**Döntés:** promptkísérletnél fix sampling, parameter experimentnél fix prompt.  
**Indok:** ne legyen confounding variable.

## Quality és output validity külön mérve
**Döntés:** a klasszifikációs minőség és az output contract validity külön metrika.  
**Indok:** egy helyes label malformed JSON-ban production integrációt ugyanúgy eltörhet.

## A mock szimulátor nem modellbizonyíték
**Döntés:** a mock output egyértelműen simulation címkét kap.  
**Indok:** a szoftvert validálja, de nem hamisít LLM-teljesítményt.

## Filesystem history adatbázis előtt
**Döntés:** run mappák/manifestelemek perzisztálása, adatbázis nélkül.  
**Indok:** egyfelhasználós portfólióalkalmazáshoz elég, könnyen inspectálható és nem overengineering.

## Natív Streamlit health check
**Döntés:** deployment probe: `/_stcore/health`.  
**Indok:** nem kell külön FastAPI service csak health check célra.
