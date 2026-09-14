# Refactor Report

## Initial state
The project already had valuable benchmark, UI, provider, visualization, and notebook functionality, but accumulated portability and maintainability debt: non-standard package layout/import workarounds, large provider/UI modules, scattered path/config assumptions, setup doing too much work, and incomplete production/deployment documentation.

## Identified problems
- package/import behavior depended on development layout;
- provider logic was overly centralized;
- some configuration/path references were duplicated;
- cache identity needed stronger guards;
- UI regressions had already occurred around token reshaping;
- setup could accidentally trigger expensive work;
- testing existed but required clearer layer classification and documentation;
- deployment/security/reproducibility expectations were not documented consistently in HU/EN.

## Architecture changes
- installable `prompt_benchmark` package under `03_src`;
- centralized `ProjectPaths` and typed config;
- provider adapters split by service with a shared base/retry contract;
- normalized `07_outputs` artifact tree;
- CLI and cross-platform launchers;
- Docker/Kubernetes assets using Streamlit health probes.

## Clean-code improvements
Reusable logic moved out of notebooks/UI where practical, names/types/docstrings improved, exceptions became more specific, secrets/path assumptions were removed, and cache/data validation boundaries became stricter.

## Testing
The current suite contains 76 passing tests across unit, integration, and smoke layers with 77% configured core-package coverage. See the bilingual test catalog for exact responsibilities.

## Known limitations
Live provider behavior, Docker runtime, and Kubernetes apply require tools/credentials/network not guaranteed in the validation environment. The bilingual Streamlit view files remain relatively large to reduce regression risk during this refactor.

## Future improvements
CI quality gates, human-reviewed adversarial data, opt-in live-provider contract tests, persistent multi-user storage, and further UI componentization.
