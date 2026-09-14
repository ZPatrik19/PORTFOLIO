# Végleges projektstruktúra

A számozás a használati és fejlesztési logikát követi, de a runtime Python-kód valódi telepíthető package marad.

```text
00_setup       -> environment/bootstrap
01_data        -> adatforrások és generált datasetek
02_notebooks   -> oktató/elemző notebookok
03_src         -> prompt_benchmark package
04_tests       -> pytest tesztek
05_scripts     -> futtató entrypointok és Streamlit UI
06_deployment  -> Kubernetes
07_outputs     -> eredmény, report, log
configs        -> prompt/provider/benchmark/pricing konfiguráció
docs           -> technikai dokumentáció
```

A Python importok nem a mappaszámozásra támaszkodnak. A projektet `pip install -e .` telepíti, így mindenhol `prompt_benchmark...` import használható.
