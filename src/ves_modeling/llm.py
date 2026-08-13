"""Provider-neutral OpenAI-compatible chat client (httpx, streaming).

Configuration:
  VES_MODELING_LLM_BASE_URL          (e.g. https://opencode.ai/zen/go/v1)
  VES_MODELING_LLM_API_KEY
  VES_MODELING_LLM_MODEL
  VES_MODELING_LLM_REASONING_EFFORT  optional (e.g. low/medium/high/max)
  VES_MODELING_LLM_MAX_TOKENS        optional (default 100000)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import httpx

logger = logging.getLogger(__name__)

# The upstream gateway throttles concurrent long SSE generations: with 2-3
# simultaneous streams one call is starved for minutes (observed 85s/156s/407s
# in a 3-way probe, and a 70+ minute no-output stall with connection pileup in
# the parallel experiment).  LLM requests are therefore serialized
# process-wide; Docker candidate execution is unaffected and stays parallel.
# This covers the current single-experiment process; future multi-process
# runners need a cross-process throttle (e.g. a file lock) for the same
# guarantee.
_LLM_REQUEST_LOCK = threading.Lock()


class OpenAICompatibleClient:
    """Minimal chat-completions client; ``complete(prompt) -> str``.

    Uses SSE streaming: long max-effort generations otherwise hit the
    gateway's non-streaming idle timeout (observed: server disconnects after
    ~100-130s on non-streaming, streaming completes in minutes).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        timeout: float = 1800.0,
        max_attempts: int = 6,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("VES_MODELING_LLM_BASE_URL", "")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("VES_MODELING_LLM_API_KEY", "")
        self.model = model or os.environ.get("VES_MODELING_LLM_MODEL", "")
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "VES_MODELING_LLM_REASONING_EFFORT", ""
        )
        self.max_tokens = max_tokens or int(
            os.environ.get("VES_MODELING_LLM_MAX_TOKENS", "100000")
        )
        self.timeout = timeout
        self.max_attempts = max_attempts
        missing = [
            name
            for name, value in (
                ("VES_MODELING_LLM_BASE_URL", self.base_url),
                ("VES_MODELING_LLM_API_KEY", self.api_key),
                ("VES_MODELING_LLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "missing LLM configuration: " + ", ".join(missing)
            )

    def complete(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        logger.info(
            "LLM request start model=%s max_tokens=%d reasoning_effort=%r",
            self.model,
            self.max_tokens,
            self.reasoning_effort or None,
        )
        # Hold the lock across the whole retry loop so at most one stream is
        # in flight per process (attempts and backoff sleeps included).
        with _LLM_REQUEST_LOCK:
            request_started = time.monotonic()
            last_error: Exception | None = None
            for attempt in range(self.max_attempts):
                try:
                    content = self._stream_completion(url, payload, headers)
                    logger.info(
                        "LLM request done model=%s attempt=%d "
                        "queue_wait=%.1fs request_elapsed=%.1fs chars=%d",
                        self.model,
                        attempt + 1,
                        request_started - started,
                        time.monotonic() - request_started,
                        len(content),
                    )
                    return content
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code < 500:
                        raise  # 4xx (e.g. auth) is not transient
                    last_error = exc
                except (httpx.TransportError, json.JSONDecodeError) as exc:
                    last_error = exc
                except RuntimeError as exc:
                    if "reasoning-only" not in str(exc):
                        raise  # non-transient client error
                    last_error = exc
                if attempt < self.max_attempts - 1:
                    logger.warning(
                        "LLM request attempt %d/%d failed (%s); retrying "
                        "in %.0fs",
                        attempt + 1,
                        self.max_attempts,
                        type(last_error).__name__,
                        15 * (attempt + 1),
                    )
                    time.sleep(15 * (attempt + 1))
            raise RuntimeError(
                "LLM request failed after "
                f"{self.max_attempts} attempts: {last_error}"
            ) from last_error

    def _stream_completion(
        self, url: str, payload: dict, headers: dict
    ) -> str:
        content_parts: list[str] = []
        with httpx.stream(
            "POST", url, json=payload, headers=headers, timeout=self.timeout
        ) as response:
            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content")
                if isinstance(chunk, str) and chunk:
                    content_parts.append(chunk)
        content = "".join(content_parts)
        if not content.strip():
            raise RuntimeError("reasoning-only: LLM returned empty content")
        return content
