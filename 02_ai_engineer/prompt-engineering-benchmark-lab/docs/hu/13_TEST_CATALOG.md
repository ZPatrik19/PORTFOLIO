# Teljes tesztkatalógus

Ez a dokumentum a repository összes pytest modulját és tesztesetét leírja. A cél nem csak annak bemutatása, hogy „vannak tesztek”, hanem hogy pontosan milyen hibamódot vagy regressziót véd minden ellenőrzés.


## Rétegek

- `unit`: gyors, izolált logika; nincs valódi külső szolgáltatás.
- `integration`: több komponens együtt; nincs fizetős/élő cloud hívás.
- `smoke`: minimális end-to-end offline pipeline.
- `external`: fenntartott marker valódi credential/network tesztekhez; alapból nem fut ilyen teszt.

## Futtatás

```bash
pytest
pytest -m unit
pytest -m integration
pytest -m smoke
pytest --cov=prompt_benchmark --cov-report=term-missing
```

## `test_advanced_strategies.py`

**Réteg:** `unit`  
**Cél:** Az advanced promptstratégiák szerződéseit, a branch-and-vote működést, a strukturált kimenetet, a prompt-izolációt és a provider-képességeket ellenőrzi.

- `test_advanced_strategy_count_and_names` — Ellenőrzi, hogy az advanced strategy registry a várt számú és stabil azonosítójú promptstratégiát adja vissza.
- `test_tree_strategy_has_three_independent_branches` — Ellenőrzi, hogy a Tree-of-Thought ihletésű branch-and-vote stratégia három független döntési ágat futtat, nem egyetlen rejtett gondolatmenetet.
- `test_full_template_uses_structured_output` — Ellenőrzi, hogy a teljes advanced template strukturált kimenetet kér, így a downstream parsing stabil szerződésre épül.
- `test_mock_extracts_delimited_ticket_not_prompt_definitions` — Megakadályozza, hogy a mock classifier a label-definíciókat vagy instrukciókat osztályozza a delimitált ticket helyett.
- `test_provider_capability_matrix_exposes_top_k_support` — Ellenőrzi, hogy a provider-képességmátrix egyértelműen jelzi a top_k támogatását.
- `test_unsupported_groq_top_k_fails_before_api_call` — Biztosítja, hogy nem támogatott Groq top_k esetén a rendszer még API-hívás előtt lokálisan hibázzon.
- `test_prompt_template_file_exists` — Ellenőrzi, hogy a UI és dokumentáció által használt újrahasznosítható prompt template fájl megtalálható a repositoryban.

## `test_benchmark_suites.py`

**Réteg:** `unit`  
**Cél:** A benchmark suite normalizálását, profilozását, kiegyensúlyozott/reprezentatív mintavételét, azonosító-megőrzését és duplikációs statisztikáit ellenőrzi.

- `test_normalize_custom_csv_schema` — Ellenőrzi, hogy a feltöltött/saját CSV adatok a kanonikus benchmark sémára normalizálódnak.
- `test_profile_reports_six_classes` — Biztosítja, hogy a dataset profil mind a hat támogatott intent osztályt megjeleníti.
- `test_balanced_sample_is_balanced` — Ellenőrzi, hogy a balanced sampling megtartja az egyenlő class-reprezentációt.
- `test_representative_subset_covers_all_labels_and_multiple_scenarios` — Ellenőrzi, hogy a pilot subset minden labelt és több scenario családot lefed, nem csak az első könnyű sorokat veszi.
- `test_normalize_preserves_unique_sample_ids` — Biztosítja, hogy a valid felhasználói sample_id-k normalizáláskor változatlanok maradnak.
- `test_profile_reports_raw_duplicate_rate_before_cleaning` — Biztosítja, hogy a duplikációs arány a nyers adaton számolódjon, még a deduplikáció előtt.

## `test_cli.py`

**Réteg:** `integration`  
**Cél:** A telepített CLI argumentumtovábbítását és a felhasználói parancsvalidációt ellenőrzi.

- `test_cli_forwards_workflow_flags` — Ellenőrzi, hogy a CLI a provider/strategy/limit és más workflow argumentumokat változtatás nélkül továbbítja.
- `test_cli_rejects_unknown_command` — Biztosítja, hogy az ismeretlen CLI parancs nem nulla exit kóddal álljon le.

## `test_config_validation.py`

**Réteg:** `unit`  
**Cél:** A konfigurációbetöltést, Pydantic-validációt és az érthető hibaüzeneteket ellenőrzi.

- `test_benchmark_config_loads_and_validates` — Betölti a benchmark YAML-t a típusos config modellen keresztül és ellenőrzi a szükséges értékeket.
- `test_missing_config_has_actionable_error` — Biztosítja, hogy hiányzó config fájl esetén érthető és javítható hibaüzenet keletkezzen.
- `test_invalid_yaml_has_actionable_error` — Biztosítja, hogy hibás YAML esetén ne átláthatatlan stack trace, hanem értelmes konfigurációs hiba jelenjen meg.

## `test_custom_prompts.py`

**Réteg:** `integration`  
**Cél:** A saját promptok renderelését, validációját, mentését/betöltését és promptérzékeny mock viselkedését ellenőrzi.

- `test_custom_prompt_renders_ticket` — Ellenőrzi, hogy a custom prompt helyesen behelyettesíti a ticket tartalmát.
- `test_custom_prompt_requires_ticket_placeholder` — Elutasítja az olyan custom template-et, amelybe a benchmark ticket nem illeszthető be.
- `test_custom_prompt_save_and_load` — Ellenőrzi a custom prompt preset veszteségmentes mentését és visszatöltését.
- `test_mock_custom_prompt_detects_advanced_components` — Biztosítja, hogy a promptérzékeny mock szimulátor felismeri a felhasználói advanced prompt komponenseket.

## `test_data_split.py`

**Réteg:** `unit`  
**Cél:** A leakage-mentes, kiegyensúlyozott és diszjunkt development/holdout/few-shot adatszétválasztást ellenőrzi.

- `test_splits_are_disjoint_and_balanced` — Ellenőrzi, hogy a development, holdout és few-shot halmazok kiegyensúlyozottak és nincs köztük átfedés.

## `test_data_validation.py`

**Réteg:** `unit`  
**Cél:** Szigorú benchmark-sémavalidációt ellenőriz még bármilyen fizetős/külső API-hívás előtt.

- `test_valid_benchmark_schema_passes` — Megerősíti, hogy a helyes benchmark DataFrame sémát a validátor elfogadja.
- `test_duplicate_ids_fail_before_api_calls` — Duplikált sample_id esetén még provider-hívás előtt leállítja a folyamatot, így nem fogy quota vagy pénz.
- `test_unknown_label_fails_before_api_calls` — Ismeretlen label esetén inference előtt hibát jelez.

## `test_expanded_challenge_set.py`

**Réteg:** `integration`  
**Cél:** A kibővített, adversarial Challenge Set méretét és scenario-diverzitását ellenőrzi.

- `test_expanded_challenge_has_eighteen_scenario_families` — Ellenőrzi, hogy a kibővített challenge benchmark mind a 18 tervezett robustness scenario családot tartalmazza.
- `test_default_generator_is_10x_scale` — Biztosítja, hogy az alapértelmezett szintetikus generátor a tervezett 10× méretű adathalmazt hozza létre.

## `test_gemini_client_adapter.py`

**Réteg:** `unit`  
**Cél:** A Gemini adapter request-összeállítását és interaction payload kompatibilitását ellenőrzi.

- `test_gemini_adapter_builds_interactions_request` — Ellenőrzi, hogy a Gemini adapter az SDK boundary számára megfelelő interactions request payloadot állít össze.

## `test_gemini_env_config.py`

**Réteg:** `unit`  
**Cél:** Az egyetlen UI-indító elvét, a közös runtime beállításokat és az elavult launcher-ek eltávolítását ellenőrzi.

- `test_root_launchers_are_explicit_and_keep_single_ui_entrypoint` — Biztosítja, hogy a root launcherek egyetlen kanonikus Streamlit entrypointra delegálnak.
- `test_unified_ui_contains_language_provider_and_api_key_controls` — Ellenőrzi, hogy a nyelv/provider/model/API-key kontrollok a közös UI runtime rétegben vannak.
- `test_obsolete_language_and_gemini_config_launchers_are_removed` — Megakadályozza az elavult HU/EN/Gemini-specifikus launcherek visszakerülését és az UX széttöredezését.

## `test_gemini_health.py`

**Réteg:** `unit`  
**Cél:** A Gemini health-check válaszfeldolgozását, tokenhasználat-kezelését és hiányzó kulcs esetét ellenőrzi.

- `test_gemini_health_parses_response_and_usage` — Ellenőrzi, hogy a Gemini health response modellválasszá és token-telemetriává alakul.
- `test_gemini_health_requires_key` — Biztosítja, hogy API kulcs nélkül a Gemini health check érthető hibával álljon le.

## `test_metrics.py`

**Réteg:** `unit`  
**Cél:** A klasszifikációs metrikákat, invalid predikciók kezelését és a kis tökéletes minták bizonytalanságát ellenőrzi.

- `test_perfect_metrics` — Ellenőrzi a metric implementációt egy tökéletes, hatosztályos predikciós példán.
- `test_confusion_table_handles_invalid_nan_predictions` — Biztosítja, hogy az invalid/NaN predikciók külön kezelhetők legyenek és ne omoljon össze a confusion matrix.
- `test_wilson_interval_keeps_small_perfect_sample_uncertain` — Biztosítja, hogy egy kis 100%-os pilot továbbra is bizonytalanságot mutasson, ne bizonyított tökéletességet sugalljon.

## `test_mock_data.py`

**Réteg:** `unit`  
**Cél:** Az offline szintetikus adatszett-generátor kiegyensúlyozottságát és minimális méretét ellenőrzi.

- `test_mock_generator_is_balanced_and_large_enough_for_default_splits` — Ellenőrzi, hogy a szintetikus forrás kiegyensúlyozott és elég nagy a konfigurált split-ekhez.

## `test_mock_prompt_sensitivity.py`

**Réteg:** `unit`  
**Cél:** A realisztikus esettípus-lefedettséget és a mock szimulátor determinisztikus quality/cost trade-offjait ellenőrzi.

- `test_mock_dataset_contains_all_realistic_case_types` — Ellenőrzi, hogy a promptérzékenységi szimulátor minden tervezett realisztikus esettípust lefed.
- `test_advanced_prompt_is_better_but_more_expensive_in_mock_simulation` — Biztosítja, hogy a mock szimulátor determinisztikus quality-versus-token/latency trade-offot modellezzen az advanced promptoknál.

## `test_openai_schema.py`

**Réteg:** `unit`  
**Cél:** A provider által kikényszerített structured output szigorú JSON Schema szerződését ellenőrzi.

- `test_structured_output_schema_is_strict_and_label_constrained` — Ellenőrzi, hogy az OpenAI-szerű JSON Schema csak a hat ismert labelt engedi és tiltja az extra mezőket.

## `test_parsing.py`

**Réteg:** `unit`  
**Cél:** A sima label- és JSON-válaszok egységes klasszifikációs szerződésbe történő parsingját ellenőrzi.

- `test_plain_label_parsing` — Ellenőrzi egy egyszerű egyszavas label válasz feldolgozását.
- `test_json_parsing` — Ellenőrzi a JSON-formátumú klasszifikációs output parsingját és validációját.

## `test_paths.py`

**Réteg:** `unit`  
**Cél:** A hordozható repository-root felderítést és az environment override működését ellenőrzi.

- `test_project_paths_point_to_repository_layout` — Ellenőrzi, hogy a ProjectPaths a megfelelő repository mappákra mutat.
- `test_project_root_env_override` — Ellenőrzi, hogy a PROMPT_BENCHMARK_ROOT explicit runtime repository rootot tud kijelölni.
- `test_path_methods_are_called_after_path_composition` — Regressziós védelem a pathlib precedenciahibára: megakadályozza az olyan kifejezéseket, mint a `root / "file.csv".exists()`, ahol az `.exists()` tévesen a stringen futna a létrehozott `Path` helyett.

## `test_playground_store.py`

**Réteg:** `unit`  
**Cél:** A Playground history robusztus betöltését ellenőrzi hiányzó vagy sérült JSONL sorok esetén.

- `test_load_playground_history_skips_malformed_rows` — Biztosítja, hogy egy sérült history sor logolódjon/kimaradjon, de a teljes Playground history betöltés ne álljon le.
- `test_load_playground_history_missing_file_returns_empty` — Biztosítja, hogy első futáskor a hiányzó Playground history üres historyként kezelődjön.

## `test_playground_token_reshape.py`

**Réteg:** `unit`  
**Cél:** Regressziós teszteket tartalmaz az A/B Playground token-átalakítására és a korábbi pandas.melt oszlopütközésre.

- `test_reshape_token_usage_allows_existing_total_tokens_column` — Regressziós teszt a korábbi pandas.melt value_name="tokens" oszlopütközésre a klasszifikációs A/B chartban.
- `test_reshape_token_usage_generation_shape` — Ellenőrzi a generatív A/B chartokhoz használt long-form token DataFrame alakját.

## `test_pricing.py`

**Réteg:** `unit`  
**Cél:** A tokenalapú request-költség számítását ellenőrzi.

- `test_cost_calculation` — Ellenőrzi az input/output tokenárazás aritmetikáját ismert várható költséggel.

## `test_prompts.py`

**Réteg:** `unit`  
**Cél:** Az alap promptstratégiák renderelését és a structured-output stratégia működését ellenőrzi.

- `test_all_strategies_render_ticket` — Biztosítja, hogy minden beépített promptstratégia rendereli a ticketet és nem veszti el a user inputot.
- `test_structured_output_strategy` — Ellenőrzi, hogy a structured-output stratégia a várt output contractot hirdeti és használja.

## `test_provider_adapters_unit.py`

**Réteg:** `unit`  
**Cél:** A provider adapterek sikeres útvonalait és credential-ellenőrzését vizsgálja élő/fizetős hívások nélkül.

- `test_openai_adapter_success` — Mockolt SDK válasszal teszteli az OpenAI adapter sikeres útvonalát és a normalizált telemetriát.
- `test_openrouter_adapter_success` — Élő hálózati hívás nélkül teszteli az OpenRouter adapter sikeres útvonalát.
- `test_groq_adapter_success` — Élő hálózati hívás nélkül teszteli a Groq adapter sikeres útvonalát.
- `test_ollama_adapter_success` — Lokálisan futó modell nélkül teszteli az Ollama adapter sikeres útvonalát.
- `test_cloud_adapters_require_keys` — Biztosítja, hogy a cloud adapterek hiányzó credential esetén még network I/O előtt hibázzanak.

## `test_provider_factory.py`

**Réteg:** `integration`  
**Cél:** A provider-registry teljességét és a credential nélküli mock kliens létrehozását ellenőrzi.

- `test_expected_providers_are_available` — Ellenőrzi, hogy Mock, Ollama, Gemini, Groq, OpenRouter és OpenAI mind regisztrálva vannak.
- `test_mock_factory_requires_no_credentials` — Biztosítja, hogy az offline mock provider secret nélkül is létrehozható.

## `test_provider_schemas.py`

**Réteg:** `unit`  
**Cél:** A Groq structured-output requestek szigorú sémáját ellenőrzi.

- `test_groq_structured_output_schema_is_strict` — Ellenőrzi, hogy a Groq schema-constrained output szigorú label enumot használ és tiltja a váratlan mezőket.

## `test_retry_policy.py`

**Réteg:** `unit`  
**Cél:** Az átmeneti hibák retry-besorolását és az autentikációs hibák fail-fast kezelését ellenőrzi.

- `test_retryable_provider_errors` — Ellenőrzi, hogy timeout/rate-limit/server jellegű átmeneti hibák retryable kategóriába kerülnek.
- `test_authentication_errors_are_not_retried` — Biztosítja, hogy hibás credential esetén ne pazaroljon retry-t és quotát a rendszer.

## `test_runner_cache.py`

**Réteg:** `integration`  
**Cél:** A benchmark cache helyességét ellenőrzi megváltozott adatok, profilok, sampling beállítások és pricing esetén.

- `test_cache_rejects_same_sample_id_with_different_text` — Megakadályozza stale cache predikció használatát, ha ugyanaz a sample_id már más tartalomra mutat.
- `test_cache_rejects_rows_outside_changed_run_profile` — Megakadályozza, hogy egy full-run cache beszennyezzen egy eltérő pilot profilt.
- `test_cache_rejects_sampling_setting_change_to_default` — Sampling beállítás változásakor invalidálja a cache-t, beleértve az alapértékre való visszaállást is.
- `test_cached_cost_is_recomputed_when_pricing_changes` — A valid predikció cache-t újrahasználja, de pricing változáskor újraszámolja a költséget.
- `test_cache_accepts_numeric_equivalent_sampling_settings` — A numerikusan ekvivalens 40 és 40.0 értékeket azonos cache-identitásként kezeli.

## `test_smoke_pipeline.py`

**Réteg:** `smoke`  
**Cél:** Minimális end-to-end offline benchmarkot futtat validációtól a promptoláson és inference-en át a metrikákig.

- `test_minimal_end_to_end_benchmark_smoke` — A legkisebb teljes offline útvonalat futtatja validált adattól prompt renderen és mock inference-en át checkpointig és evaluációig.

## `test_ui_language_switch.py`

**Réteg:** `integration`  
**Cél:** A kétnyelvű egy-UI architektúrát, közös runtime beállításokat és HU/EN funkcióparitást ellenőrzi.

- `test_bilingual_ui_launcher_exists_and_loads_both_views` — Biztosítja, hogy a közös launcher mind a magyar, mind az angol nézetet be tudja tölteni.
- `test_language_views_do_not_reset_page_config` — Megakadályozza, hogy a nyelvi view modulok újrainicializálják a globális Streamlit page configot.
- `test_ui_bat_delegates_to_single_project_launcher` — Ellenőrzi, hogy a legacy RUN_UI.bat a kanonikus project launcherre delegál.
- `test_runtime_controls_live_only_in_shared_launcher` — Megakadályozza a provider/model/API-key kontrollok duplikálását a nyelvspecifikus view-kban.
- `test_both_languages_expose_custom_prompt_ui` — Ellenőrzi, hogy a HU és EN felület is elérhetővé teszi a custom prompt funkciót.

## `test_workflow_history.py`

**Réteg:** `integration`  
**Cél:** A tartós benchmark history és Playground preset mentés-visszatöltés körét ellenőrzi.

- `test_run_history_round_trip` — Ellenőrzi, hogy a benchmark run manifest és eredmények menthetők/visszatölthetők metadata-vesztés nélkül.
- `test_generation_preset_round_trip` — Ellenőrzi, hogy a generatív Playground preset veszteségmentesen menthető és visszatölthető.
