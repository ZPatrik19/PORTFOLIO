# FastAPI Service Design

FastAPI uses Python type hints and validation models to define request and response contracts. Pydantic schemas reduce ambiguity at API boundaries. Health and readiness endpoints should have different semantics: health asks whether the process is alive, while readiness asks whether dependencies and required state are usable.
