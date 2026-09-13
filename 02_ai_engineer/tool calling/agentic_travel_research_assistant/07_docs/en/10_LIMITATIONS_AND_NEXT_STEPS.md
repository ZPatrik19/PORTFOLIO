# Limitations and next steps

This chapter clearly separates current system capability from genuinely useful next-stage engineering work.

## 1. Current limitations

- Hotel/restaurant/attraction inventories are synthetic rather than live availability.
- The domain is limited to 60 cities and predefined capabilities.
- The ML router is a classical TF-IDF model rather than a semantic encoder.
- Deterministic argument extraction remains heuristic.
- The workflow does not use a production queue or distributed tracing.
- Live API integration is limited to weather and currency.
- Chat answer quality depends on routing mode; offline modes do not provide general-purpose generative reasoning.

## 2. Meaningful next steps

1. Real hotel/flight/POI provider integration with explicit API contracts.
2. Async parallel execution for independent tools.
3. DAG/dependency executor for complex workflows.
4. Semantic router or compact transformer compared against the TF-IDF baseline.
5. OpenTelemetry-compatible tracing and production monitoring.
6. Tool-level timeout/retry/circuit-breaker policies.
7. Human approval before side-effecting tools.
8. Secrets management and per-tool permissions.
9. Adversarial/prompt-injection and schema-failure evaluation.
10. Separate backend/API service rather than coupling runtime to Streamlit.

## 3. What should not be added merely for complexity?

A multi-agent framework, Kubernetes or a vector database does not automatically improve this project. Such components are justified only by a concrete requirement such as scaling, shared state, semantic retrieval or side-effecting workflows.

## Conclusion

The current repository is a coherent portfolio-scale system. The next iteration should solve a real new requirement rather than add technology for its own sake.
