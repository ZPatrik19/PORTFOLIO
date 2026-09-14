# Design Decisions

## Installable package instead of `sys.path` hacks
**Decision:** core logic lives in `03_src/prompt_benchmark` and is installed with `pip install -e .`.  
**Reason:** the same import behavior must work from CLI, notebooks, tests, Docker, and Kubernetes.

## Repository-relative paths
**Decision:** use `ProjectPaths` and `pathlib`.  
**Reason:** remove developer-machine coupling and make Windows/Linux/container execution consistent.

## Separate prompt benchmark from decoding sweep
**Decision:** prompt experiments hold sampling settings fixed; parameter experiments hold prompt fixed.  
**Reason:** avoid confounding variables.

## Separate quality from output validity
**Decision:** measure classification quality and contract validity independently.  
**Reason:** a correct label inside malformed JSON can still break a production integration.

## Mock simulator is not model evidence
**Decision:** mock output is explicitly labelled simulation.  
**Reason:** it validates software behavior without misrepresenting LLM performance.

## Filesystem history before database
**Decision:** persist run folders/manifests instead of adding a database prematurely.  
**Reason:** adequate for a single-user portfolio app, simpler to inspect and version.

## Native Streamlit health check
**Decision:** use `/_stcore/health` for deployment probes.  
**Reason:** no need to invent a second FastAPI service only for health checking.
