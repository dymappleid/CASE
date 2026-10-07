"""Thin host-facing helpers for running CASE with VideoSeek and AVP."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .online import OnlineController


MODEL_BACKENDS = {
    "qwen35": {
        "checkpoint": "Qwen/Qwen3.5-35B-A3B-FP8",
        "revision": "9d1823d2dee688a6b25e77009dc727688c44936e",
        "context_length": 65536,
    },
    "glm46v": {
        "checkpoint": "zai-org/GLM-4.6V-Flash",
        "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "context_length": 65536,
    },
    "internvl35": {
        "checkpoint": "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "revision": "647fbc96f5fc9798f9f5811ddf622f07a09ccb9f",
        "context_length": 32768,
    },
}

# Native settings used for VideoSeek and AVP evaluation.
VIDEOSEEK_DEFAULTS = {
    "seed": 42,
    "temperature": 1.0,
    "max_tokens": 1024,
    "max_steps": 20,
    "tools": ("overview", "skim", "focus", "answer"),
    "frame_sampling_factor": 4,
    "overview_base": 8,
    "skim_base": 2,
    "focus_base": 2,
    "max_visual_actions_per_turn": 1,
}

AVP_DEFAULTS = {
    "seed": 42,
    "max_rounds": 5,
    "confidence_threshold": 0.95,
    "max_tokens": 1024,
    "frame_caps": {"low": 20, "medium": 20, "high": 20},
}


def host_defaults(host: str, model: str) -> dict[str, Any]:
    """Return native host settings without deployment-specific values."""

    if model not in MODEL_BACKENDS:
        raise KeyError(f"unsupported model: {model}")
    if host == "videoseek":
        return deepcopy(VIDEOSEEK_DEFAULTS)
    if host == "avp":
        settings = deepcopy(AVP_DEFAULTS)
        settings["context_retry"] = model == "internvl35"
        if model == "internvl35":
            settings["context_retry_count"] = 1
            settings["context_retry_frame_ratio"] = 0.70
        return settings
    raise KeyError(f"unsupported host: {host}")


class HostAdapter:
    """Narrow hook from a native host loop into an ``OnlineController``."""

    def __init__(self, host: str, model: str, controller: OnlineController) -> None:
        self.host = host
        self.model = model
        self.controller = controller
        self.native_settings = host_defaults(host, model)

    def start(
        self,
        *,
        question: str,
        letters: Sequence[str],
        duration_group: str,
        video_duration_sec: float,
        has_subtitle: bool,
    ) -> None:
        self.controller.start(
            question=question,
            letters=letters,
            duration_group=duration_group,
            video_duration_sec=video_duration_sec,
            has_subtitle=has_subtitle,
        )

    def evaluate(
        self,
        *,
        evidence: str,
        latest_tool: str | None,
        consumed_cost: Mapping[str, Any],
        decision_sunk_cost: Mapping[str, Any],
        native_prefix_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.controller.step(
            evidence=evidence,
            latest_tool=latest_tool,
            consumed_cost=consumed_cost,
            decision_sunk_cost=decision_sunk_cost,
            native_prefix_messages=native_prefix_messages,
        )
