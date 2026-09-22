"""OpenRouter API inference backend.

Drop-in replacement for MLXRunner / VLLMRunner.  Sends chat-completion
requests to OpenRouter's OpenAI-compatible endpoint and extracts label-
token log-probabilities for the constrained two-way softmax used by the
rest of the pipeline (src/inference/llm_runner.py::get_confidence).

The API handles chat templating internally, so this runner accepts
(system, user) message pairs directly -- no ChatFormatter needed. A
lightweight shim is provided so callers that expect `runner.chat_formatter()`
still work without branching.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import requests

from src.inference.llm_runner import INVALID, PredictionResult, get_confidence


_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class _PassthroughFormatter:
    """Shim so callers that do `fmt = runner.chat_formatter(); fmt.render(sys, usr)`
    get back a JSON-encoded message list rather than a chat-templated string.
    The runner's batch_predict then detects this and reconstructs messages.
    """

    @staticmethod
    def render(system: str, user: str) -> str:
        import json
        return json.dumps([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])


class OpenRouterRunner:
    """Calls OpenRouter's chat-completions API for classification."""

    def __init__(
        self,
        model: str = "qwen/qwen3-8b:free",
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        requests_per_second: float = 5.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key: set OPENROUTER_API_KEY env var or pass api_key="
            )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self._last_request_time = 0.0

    def chat_formatter(self) -> _PassthroughFormatter:
        return _PassthroughFormatter()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _call_api(self, messages: list[dict], logprobs: bool = True) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 20

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = requests.post(
                    _API_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  [OpenRouter] rate-limited, waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  [OpenRouter] request error: {e}, retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        return {}

    def _extract_logprobs(
        self, response: dict, label_tokens: tuple[str, str]
    ) -> dict[str, float] | None:
        """Try to extract label-token logprobs from the API response."""
        try:
            choice = response["choices"][0]
            lp_content = choice.get("logprobs", {})
            if not lp_content:
                return None
            top_logprobs_list = lp_content.get("content", [])
            if not top_logprobs_list:
                return None
            first_token_logprobs = top_logprobs_list[0].get("top_logprobs", [])
            if not first_token_logprobs:
                return None

            logprobs_dict = {}
            for entry in first_token_logprobs:
                token = entry.get("token", "").strip()
                lp = entry.get("logprob", -100)
                if token in label_tokens:
                    logprobs_dict[token] = lp

            if logprobs_dict:
                return logprobs_dict
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def _extract_generated_token(self, response: dict) -> str:
        """Fall back to the generated text when logprobs aren't available."""
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return INVALID

    def predict_one(
        self, messages: list[dict], label_tokens: tuple[str, str]
    ) -> PredictionResult:
        response = self._call_api(messages, logprobs=True)

        logprobs_dict = self._extract_logprobs(response, label_tokens)
        if logprobs_dict:
            return get_confidence(logprobs_dict, label_tokens)

        token = self._extract_generated_token(response)
        if token in label_tokens:
            idx = label_tokens.index(token)
            p0 = 1.0 if idx == 0 else 0.0
            p1 = 1.0 - p0
            return PredictionResult(
                prediction=token,
                confidence=1.0,
                p0=p0,
                p1=p1,
                logprob_0=0.0 if idx == 0 else -100.0,
                logprob_1=0.0 if idx == 1 else -100.0,
            )
        return PredictionResult(
            prediction=INVALID, confidence=0.0,
            p0=0.0, p1=0.0, logprob_0=-100.0, logprob_1=-100.0,
        )

    def batch_predict(
        self, prompts: list[str], label_tokens: tuple[str, str]
    ) -> list[PredictionResult]:
        """Match the MLXRunner/VLLMRunner interface.

        `prompts` are either:
          (a) JSON-encoded message lists (from _PassthroughFormatter.render), or
          (b) raw chat-templated strings (from ChatFormatter.render).

        Case (a) is the expected path with this runner. Case (b) falls back
        to wrapping the string as a single user message.
        """
        import json

        results = []
        total = len(prompts)
        for i, prompt in enumerate(prompts):
            try:
                messages = json.loads(prompt)
                if not isinstance(messages, list):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                messages = [{"role": "user", "content": prompt}]

            if (i + 1) % 10 == 0 or i == 0 or i == total - 1:
                print(f"  [OpenRouter] {i+1}/{total}...")

            results.append(self.predict_one(messages, label_tokens))

        return results
