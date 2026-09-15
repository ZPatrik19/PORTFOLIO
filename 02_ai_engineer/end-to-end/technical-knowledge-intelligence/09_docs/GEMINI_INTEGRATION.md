# Gemini Integration

The project uses `google-genai`, with the Interactions API for generation/function calling and `models.embed_content` for embeddings. Model names and dimensions live in `config.yaml`; the API key is read from `.env` only.

The integration includes retries with exponential backoff. When Gemini is unavailable, retrieval still works and the platform returns an explicitly labeled extractive fallback instead of crashing or fabricating an API result.
