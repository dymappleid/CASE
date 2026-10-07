"""Minimal online CASE controller for OpenAI-compatible multimodal backends."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import (
    FrozenPolicy,
    activation_enabled,
    canonical_cost,
    feature_vector,
)
from .fit import checkpoint_group, temperature_scaled_distribution


def _extract_choice(text: str | None, letters: Sequence[str]) -> str:
    value = str(text or "").strip().upper()
    allowed = {letter.upper() for letter in letters}
    if value in allowed:
        return value
    match = re.search(r"\b([A-E])\b", value)
    if match and match.group(1) in allowed:
        return match.group(1)
    raise RuntimeError(f"backend returned no valid choice: {text!r}")


class OpenAIChoiceClient:
    """Dependency-free client for OpenAI-compatible choice-constrained calls."""

    def __init__(self, api_base: str, served_model: str, timeout: float = 900.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.served_model = served_model.removeprefix("openai/")
        self.timeout = float(timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("CASE_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        request = Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"model request failed: {error}; response={detail[:1000]}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"model request failed: {error}") from error
        return body, (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _cost(body: Mapping[str, Any], latency_ms: float) -> dict[str, float]:
        usage = body.get("usage") or {}
        cost = canonical_cost(
            {
                "text_input_token": usage.get("prompt_tokens", 0),
                "text_output_token": usage.get("completion_tokens", 0),
                "llm_calls": 1,
            }
        )
        cost["latency_ms"] = float(latency_ms)
        return cost

    def probe(
        self, evidence: str, letters: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        payload: dict[str, Any] = {
            "model": self.served_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the multiple-choice question using only the supplied visible "
                        "evidence. Return exactly one allowed uppercase option letter."
                    ),
                },
                {"role": "user", "content": evidence},
            ],
            "max_tokens": 1,
            "seed": 42,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            "structured_outputs": {"choice": list(letters)},
        }
        if self.served_model.lower().startswith("qwen3"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body, latency = self._post(payload)
        content = (
            ((body.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
            or []
        )
        if not content:
            raise RuntimeError("probe response has no token log-probabilities")
        token_rows = [content[0], *((content[0].get("top_logprobs") or []))]
        by_letter: dict[str, list[float]] = {letter: [] for letter in letters}
        for row in token_rows:
            token = str(row.get("token") or "").strip().upper()
            if token in by_letter:
                by_letter[token].append(float(row["logprob"]))
        missing = [letter for letter, values in by_letter.items() if not values]
        if missing:
            raise RuntimeError("probe omitted option logits: " + ", ".join(missing))
        return (
            {letter: max(values) for letter, values in by_letter.items()},
            self._cost(body, latency),
        )

    def finalize(
        self,
        messages: list[dict[str, Any]],
        question: str,
        letters: Sequence[str],
        max_tokens: int = 1024,
    ) -> tuple[str, str, dict[str, float]]:
        payload: dict[str, Any] = {
            "model": self.served_model,
            "messages": [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nPlease directly provide the final answer "
                        "using the required format in the question."
                    ),
                },
            ],
            "max_tokens": int(max_tokens),
            "seed": 42,
            "temperature": 0.0,
            "structured_outputs": {"choice": list(letters)},
        }
        if self.served_model.lower().startswith("qwen3"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body, latency = self._post(payload)
        message = (body.get("choices") or [{}])[0].get("message") or {}
        raw = str(message.get("content") or message.get("reasoning") or "").strip()
        return _extract_choice(raw, letters), raw, self._cost(body, latency)


class OnlineController:
    """Stateful, label-blind STOP/CONTINUE controller for one frozen policy."""

    def __init__(self, policy: FrozenPolicy, client: OpenAIChoiceClient) -> None:
        self.policy = policy
        self.client = client
        self.reset()

    def reset(self) -> None:
        self.duration_group = "short"
        self.video_duration_sec = 0.0
        self.has_subtitle = False
        self.question = ""
        self.letters: tuple[str, ...] = ()
        self.checkpoints: list[dict[str, Any]] = []

    def start(
        self,
        *,
        question: str,
        letters: Sequence[str],
        duration_group: str,
        video_duration_sec: float,
        has_subtitle: bool,
    ) -> None:
        self.reset()
        self.question = question
        self.letters = tuple(str(letter).upper() for letter in letters)
        self.duration_group = duration_group
        self.video_duration_sec = float(video_duration_sec)
        self.has_subtitle = bool(has_subtitle)

    @property
    def enabled(self) -> bool:
        return activation_enabled(self.duration_group, self.policy.operating_point.activation)

    def step(
        self,
        *,
        evidence: str,
        latest_tool: str | None,
        consumed_cost: Mapping[str, Any],
        decision_sunk_cost: Mapping[str, Any],
        native_prefix_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "stop": False, "costs": []}

        index = len(self.checkpoints)
        logits, probe_cost = self.client.probe(evidence, self.letters)
        distribution = temperature_scaled_distribution(
            logits, self.policy.probe_temperatures[checkpoint_group(index)]
        )
        checkpoint = {
            "latest_tool": latest_tool or "none",
            "evidence_char_count": len(evidence),
            "consumed_cost": canonical_cost(consumed_cost),
            "decision_sunk_cost": canonical_cost(decision_sunk_cost),
            "probe_cost": probe_cost,
            "calibrated_distribution": distribution,
        }
        self.checkpoints.append(checkpoint)
        duration_feature = (
            self.video_duration_sec
            if self.policy.feature_video_duration_sec is None
            else self.policy.feature_video_duration_sec
        )
        trajectory = {
            "duration_group": self.duration_group,
            "video_duration_sec": duration_feature,
            "has_subtitle": self.has_subtitle,
            "checkpoints": self.checkpoints,
        }
        score = float(
            self.policy.ridge_model.predict(feature_vector(trajectory, index)[None, :])[0]
        )
        stop = score <= self.policy.operating_point.boundary
        result: dict[str, Any] = {
            "enabled": True,
            "stop": stop,
            "checkpoint_index": index,
            "score": score,
            "boundary": self.policy.operating_point.boundary,
            "calibrated_distribution": distribution,
            "costs": [probe_cost],
        }
        if stop:
            answer, raw, finalizer_cost = self.client.finalize(
                native_prefix_messages, self.question, self.letters
            )
            result.update(answer=answer, raw_answer=raw)
            result["costs"].append(finalizer_cost)
        return result
