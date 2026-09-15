# Context Engineering

`ContextBuilder` enforces a character budget, removes near-duplicates and caps chunks per document. This prevents one highly repetitive source from consuming the full prompt and supports multi-document comparison. Source metadata is rendered inside explicit source boundaries so citation IDs are machine-checkable.
