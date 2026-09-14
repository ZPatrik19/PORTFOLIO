# Free / Low-cost LLM Setup

## Offline software validation
Use `mock` to validate the entire pipeline with zero credentials and zero model cost. It is a simulator, not an LLM benchmark.

## Local real model
Use Ollama for a real local model without per-request cloud billing. Cost shifts to local CPU/GPU/RAM/time.

## Cloud free tiers
Gemini, Groq, and OpenRouter can expose free-tier/free-model options depending on current account, region, quotas, and provider policy. These terms change over time; the repository intentionally does not promise a permanent free quota.

## Recommended workflow
1. Mock smoke test.
2. Ollama or a cloud free-tier pilot on 12–36 representative samples.
3. Standard 120-sample run.
4. Larger run only after connection, parsing, quota, and cost behavior are understood.

Never place API keys in source or screenshots intended for GitHub.
