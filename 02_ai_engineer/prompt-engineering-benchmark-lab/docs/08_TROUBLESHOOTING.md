# Troubleshooting

## Python command not found
Run `run_project.bat` on Windows; it can use `winget` to install Python 3.12 when available. On Linux/macOS install Python 3.10–3.14 first.

## `ModuleNotFoundError: prompt_benchmark`
Install the repository package:

```bash
python -m pip install -e . --no-deps
```

The project no longer relies on `sys.path.append` hacks.

## Missing API key
Mock and Ollama do not require a cloud key. For Gemini/Groq/OpenRouter/OpenAI enter the key in the UI sidebar or create a local ignored `.env` from `.env.example`.

## API authentication failure
Do not retry blindly. Verify that the key belongs to the selected provider and model access is enabled. Authentication failures are treated as non-transient by the retry policy.

## API quota / 429
Reduce benchmark profile/sample count or wait for quota reset. Transient rate-limit failures use bounded exponential backoff, but persistent quota exhaustion still surfaces as an error.

## Network timeout / provider 5xx
The common provider layer retries bounded transient errors. Check local firewall/DNS and provider status if retries are exhausted.

## Benchmark cache mismatch
If a cached strategy file was created with another model, dataset or decoding setting, the runner rejects it. Use a different output/run history or explicitly rerun with `--force`.

## Empty / malformed CSV upload
Required fields are `text` and `label`/`true_label`; final benchmark data must have unique `sample_id`, non-empty text and supported labels.

## Port 8501 is already in use
Stop the existing Streamlit process or start via CLI with another port:

```bash
prompt-benchmark ui --port 8502
```

## Docker build fails
Verify network access to PyPI and enough disk space. Secrets are not needed at build time.

## Kubernetes `CrashLoopBackOff`

```bash
kubectl -n prompt-benchmark describe pod <pod>
kubectl -n prompt-benchmark logs <pod>
```

Check image availability, memory limits and malformed environment values.
