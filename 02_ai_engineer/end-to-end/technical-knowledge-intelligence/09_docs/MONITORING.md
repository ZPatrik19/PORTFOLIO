# Monitoring

Every orchestrated request receives request and trace IDs. Structured JSONL plus SQLite capture query type, retrieval scores, context size, component latencies, citations, tool usage, validation state and feedback. The Streamlit monitoring page exposes aggregate health and time series.

The drift module compares recent and previous windows for retrieval score, no-answer rate and retrieved chunk count. Thresholds are simple and explainable; production alerting would add seasonality, confidence intervals and external metric backends.
