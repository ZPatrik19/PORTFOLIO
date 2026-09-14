# Troubleshooting

## Python not found
Install Python 3.10–3.14 and reopen the terminal. On Windows, the setup script can use the `py` launcher when available.

## `ModuleNotFoundError: prompt_benchmark`
Run `python -m pip install -e .` from repository root, or use the provided launchers.

## Missing API key
Use Mock/Ollama without cloud credentials, or enter the cloud API key in the UI/runtime environment. Do not hard-code it into source files.

## Provider authentication error
Check the key, provider selection, model name, project/billing/free-tier status, and whether the key has the required API access. Authentication errors are intentionally not retried.

## Quota / 429
Reduce benchmark size, wait for quota reset, select a free/local provider, or use resumable history/checkpointing. Retry logic is only appropriate for transient rate-limit responses.

## Playground chart error
Run the latest version: token reshaping has a regression test covering the former pandas `melt` column-name collision.

## Port 8501 already in use
Stop the previous Streamlit process or start the CLI with `prompt-benchmark ui --port <other-port>`.

## Docker build fails
Check Docker availability, network access to package indexes, and that no local `.venv`/large outputs are being copied. `.dockerignore` should exclude them.

## Kubernetes CrashLoopBackOff
Inspect `kubectl logs`, `kubectl describe pod`, environment variables/secrets, memory limits, and Streamlit startup errors.
