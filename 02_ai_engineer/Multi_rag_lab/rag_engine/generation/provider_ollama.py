from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests


class OllamaGenerationError(RuntimeError):
    """Raised when the local Ollama runtime cannot complete a generation."""


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:4b",
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        max_retries: int = 1,
        num_predict: int = 384,
        temperature: float = 0.15,
        keep_alive: str = "5m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.connect_timeout = max(1.0, float(connect_timeout))
        self.read_timeout = max(30.0, float(read_timeout))
        self.max_retries = max(0, int(max_retries))
        self.num_predict = max(32, int(num_predict))
        self.temperature = float(temperature)
        self.keep_alive = keep_alive
        self.last_metrics: dict[str, float | int | str] = {}

    @staticmethod
    def _duration_ms(value: Any) -> float:
        try:
            return float(value or 0.0) / 1_000_000.0
        except TypeError, ValueError:
            return 0.0

    def _stream_once(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        extract_piece: Callable[[dict[str, Any]], str],
        num_predict: int,
        attempt: int,
    ) -> str:
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        final_event: dict[str, Any] = {}

        with requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            stream=True,
            timeout=(self.connect_timeout, self.read_timeout),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise OllamaGenerationError(str(event["error"]))
                piece = extract_piece(event)
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                if event.get("done"):
                    final_event = event

        total_ms = (time.perf_counter() - started) * 1000.0
        eval_count = int(final_event.get("eval_count") or 0)
        eval_duration_ns = float(final_event.get("eval_duration") or 0.0)
        tokens_per_second = eval_count / max(eval_duration_ns / 1_000_000_000.0, 1e-9) if eval_count else 0.0
        self.last_metrics = {
            "ttft_ms": ((first_token_at - started) * 1000.0) if first_token_at is not None else total_ms,
            "total_ms": total_ms,
            "output_tokens": eval_count,
            "prompt_tokens": int(final_event.get("prompt_eval_count") or 0),
            "tokens_per_second": tokens_per_second,
            "load_ms": self._duration_ms(final_event.get("load_duration")),
            "prompt_eval_ms": self._duration_ms(final_event.get("prompt_eval_duration")),
            "generation_eval_ms": self._duration_ms(final_event.get("eval_duration")),
            "done_reason": str(final_event.get("done_reason") or ""),
            "attempt": attempt,
            "num_predict": num_predict,
            "endpoint": endpoint,
        }
        return "".join(pieces).strip()

    def _run_with_retry(self, factory: Callable[[int, int], str]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_predict = self.num_predict if attempt == 0 else min(self.num_predict, 192)
            try:
                return factory(retry_predict, attempt + 1)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.RequestException as exc:
                raise OllamaGenerationError(f"Ollama HTTP hiba: {exc}") from exc
        raise OllamaGenerationError(
            "Az Ollama generálás időtúllépés miatt nem fejeződött be. "
            f"Modell: {self.model_name}; read timeout: {self.read_timeout:.0f}s. "
            "A lokális modell lehet még betöltés alatt vagy kevés a rendelkezésre álló VRAM. "
            "Próbáld a Qwen Low Memory profilt, kisebb kontextuskeretet vagy futtasd újra a kérést."
        ) from last_error

    def generate(self, prompt: str) -> str:
        def call(num_predict: int, attempt: int) -> str:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {"num_predict": num_predict, "temperature": self.temperature},
            }
            return self._stream_once(
                endpoint="/api/generate",
                payload=payload,
                extract_piece=lambda event: str(event.get("response", "")),
                num_predict=num_predict,
                attempt=attempt,
            )

        return self._run_with_retry(call)

    def generate_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Use Ollama chat roles so language/grounding instructions have system priority."""

        def call(num_predict: int, attempt: int) -> str:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {"num_predict": num_predict, "temperature": self.temperature},
            }
            return self._stream_once(
                endpoint="/api/chat",
                payload=payload,
                extract_piece=lambda event: str((event.get("message") or {}).get("content", "")),
                num_predict=num_predict,
                attempt=attempt,
            )

        return self._run_with_retry(call)

    def health(self) -> dict[str, object]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(3, 5))
            response.raise_for_status()
            models = [str(m.get("name", "")) for m in response.json().get("models", [])]
            installed = any(name == self.model_name or name.startswith(f"{self.model_name}:") for name in models)
            return {
                "ok": True,
                "provider": self.name,
                "model": self.model_name,
                "installed": installed,
                "models": models,
                "read_timeout_seconds": self.read_timeout,
                "num_predict": self.num_predict,
            }
        except requests.RequestException as exc:
            return {"ok": False, "provider": self.name, "model": self.model_name, "error": str(exc)}
