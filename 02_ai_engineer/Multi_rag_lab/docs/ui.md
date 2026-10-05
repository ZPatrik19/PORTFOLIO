# Streamlit UI

A UI presentation layer, a core pipeline a `rag_engine/` csomagban marad.

```text
ui/
├── app.py
├── pages/
├── components/
└── charts/
```

A fő oldalak lefedik a dokumentumkezelést, chunkingot, embeddinget, vector search-t, RAG playgroundot, összehasonlítást, evaluationt, pipeline matrixot, performance labot és runtime állapotot.

Az UI komponensek nem indíthatnak automatikusan ismételt drága LLM hívásokat pusztán Streamlit rerun miatt; erre session-state alapú vezérlés szolgál.
