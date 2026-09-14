# CLI referencia

## Telepített parancs
```bash
prompt-benchmark --help
prompt-benchmark prepare-data --source mock
prompt-benchmark benchmark --provider mock --strategy p0_zero_shot --limit 12
prompt-benchmark ablation --provider mock
prompt-benchmark parameter-sweep --provider mock
prompt-benchmark report --provider mock
prompt-benchmark smoke
prompt-benchmark ui --port 8501
```

A workflow-specifikus flag-ek a meglévő scriptekhez továbbítódnak, így a benchmark argumentum parsingnak egy source of truthja marad.

## Közvetlen scriptek
Az `05_scripts/` számozott scriptjei oktatási/debug célból megmaradnak. Normál használatra CLI/UI ajánlott; közvetlen script akkor hasznos, ha egy konkrét pipeline lépést kell reprodukálni.
