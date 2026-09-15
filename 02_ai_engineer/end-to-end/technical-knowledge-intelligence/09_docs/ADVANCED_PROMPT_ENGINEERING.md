# Advanced Prompt Engineering Layer

Prompt optimization is a separate stage before query understanding and retrieval.

```text
Raw user question
  ↓
Language / clarity cleanup
  ↓
Selected advanced prompt profile
  ↓
Optional Gemini refinement
  ↓
Optimized query/task
  ↓
Intent classification + retrieval
```

Available profiles:

- **RAG Grounded** — emphasizes evidence, citations and abstention.
- **CO-STAR** — Context, Objective, Style, Tone, Audience, Response.
- **CRISPE** — role/context/task/constraints/expected output scaffold.
- **Technical Deep Dive** — expands a short technical query into architecture, mechanisms, trade-offs and production implications.
- **Learning / Tutor** — prerequisites → intuition → concepts → math → implementation → mistakes → quiz → reading.

The optimizer is forbidden from answering the question or adding facts. It only makes the task and expected response more explicit. The UI shows **Before optimization** and **After optimization**, and the final assembled model prompt remains inspectable in Developer mode.
