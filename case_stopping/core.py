"""Core CASE state representation, decision-gap target, and ridge model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping

import numpy as np

TOOLS = ("overview", "skim", "focus", "answer", "none")
FEATURE_NAMES = (
    "option_count",
    "rank_probability_1",
    "rank_probability_2",
    "rank_probability_3",
    "normalized_entropy",
    "top1_top2_margin",
    "aligned_total_variation",
    "top_set_changed",
    "checkpoint_index",
    "log_consumed_model_tokens",
    "log_evidence_char_count",
    "log_video_duration_sec",
    "has_subtitle",
    *(f"latest_tool_{tool}" for tool in TOOLS),
)


@dataclass(frozen=True)
class OperatingPoint:
    """One frozen CASE operating point."""

    cost_weight: float
    boundary: float
    activation: str = "medium_long"

    def __post_init__(self) -> None:
        if self.activation not in {"medium_long", "long_only", "all"}:
            raise ValueError(f"unsupported activation: {self.activation}")


def activation_enabled(duration_group: str, activation: str) -> bool:
    if activation == "all":
        return True
    if activation == "medium_long":
        return duration_group in {"medium", "long"}
    if activation == "long_only":
        return duration_group == "long"
    raise ValueError(f"unsupported activation: {activation}")


def model_tokens(cost: Mapping[str, Any]) -> float:
    """Return text + visual tokens from the public canonical cost schema."""

    text = cost.get("text_token")
    if text is None:
        text = float(cost.get("text_input_token", 0) or 0) + float(
            cost.get("text_output_token", 0) or 0
        )
    return float(text or 0) + float(cost.get("visual_token", 0) or 0)


def canonical_cost(cost: Mapping[str, Any]) -> dict[str, float]:
    """Normalize one host/controller cost record to CASE's public schema."""

    text_input = float(cost.get("text_input_token", 0) or 0)
    text_output = float(cost.get("text_output_token", 0) or 0)
    text = cost.get("text_token")
    if text is None:
        text = text_input + text_output
    return {
        "text_input_token": text_input,
        "text_output_token": text_output,
        "text_token": float(text or 0),
        "visual_token": float(cost.get("visual_token", 0) or 0),
        "llm_calls": float(cost.get("llm_calls", 0) or 0),
        "tool_calls": float(cost.get("tool_calls", 0) or 0),
    }


def _distribution(checkpoint: Mapping[str, Any]) -> dict[str, float]:
    raw = checkpoint.get("calibrated_distribution")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no calibrated option distribution")
    values = {str(key): float(value) for key, value in raw.items()}
    if not 3 <= len(values) <= 5:
        raise ValueError("CASE expects 3, 4, or 5 answer choices")
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("invalid option probability")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ValueError("option probabilities must sum to one")
    return values


def _top_set(distribution: Mapping[str, float]) -> frozenset[str]:
    maximum = max(distribution.values())
    return frozenset(
        key
        for key, value in distribution.items()
        if math.isclose(value, maximum, abs_tol=1e-12)
    )


def feature_vector(trajectory: Mapping[str, Any], index: int) -> np.ndarray:
    """Return the 18-dimensional causal state representation used by CASE."""

    checkpoint = trajectory["checkpoints"][index]
    distribution = _distribution(checkpoint)
    previous = (
        _distribution(trajectory["checkpoints"][index - 1])
        if index
        else distribution
    )
    if set(previous) != set(distribution):
        raise ValueError("adjacent checkpoints must use the same option alphabet")

    ordered = sorted(distribution.values(), reverse=True)
    entropy = -sum(
        max(value, 1e-12) * math.log(max(value, 1e-12))
        for value in distribution.values()
    ) / math.log(len(distribution))
    total_variation = 0.5 * sum(
        abs(distribution[key] - previous[key]) for key in distribution
    )
    latest_tool = str(checkpoint.get("latest_tool") or "none")
    consumed = checkpoint["consumed_cost"]

    return np.asarray(
        [
            float(len(distribution)),
            ordered[0],
            ordered[1],
            ordered[2],
            entropy,
            ordered[0] - ordered[1],
            total_variation,
            float(_top_set(distribution) != _top_set(previous)),
            float(index),
            math.log1p(model_tokens(consumed)),
            math.log1p(float(checkpoint.get("evidence_char_count", 0) or 0)),
            math.log1p(float(trajectory.get("video_duration_sec", 0) or 0)),
            float(bool(trajectory.get("has_subtitle"))),
            *(float(latest_tool == tool) for tool in TOOLS),
        ],
        dtype=float,
    )


class RidgeModel:
    """Small weighted ridge model used for the continuous decision gap."""

    def __init__(self, alpha: float = 10.0) -> None:
        if alpha < 0:
            raise ValueError("ridge alpha must be non-negative")
        self.alpha = float(alpha)

    def fit(
        self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray
    ) -> "RidgeModel":
        self.mean = np.average(features, axis=0, weights=weights)
        self.target_mean = float(np.average(targets, weights=weights))
        self.scale = np.sqrt(
            np.average((features - self.mean) ** 2, axis=0, weights=weights)
        )
        self.scale[self.scale < 1e-8] = 1.0
        normalized = (features - self.mean) / self.scale
        matrix = (normalized.T * weights) @ normalized
        matrix += self.alpha * np.eye(normalized.shape[1])
        target = targets - self.target_mean
        self.coefficients = np.linalg.solve(
            matrix, (normalized.T * weights) @ target
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.target_mean + (
            (features - self.mean) / self.scale
        ) @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "target_mean": self.target_mean,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RidgeModel":
        model = cls(float(payload["alpha"]))
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.scale = np.asarray(payload["scale"], dtype=float)
        model.target_mean = float(payload["target_mean"])
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        return model


@dataclass(frozen=True)
class FrozenPolicy:
    """A fitted CASE controller ready for offline or online application."""

    host: str
    model: str
    operating_point: OperatingPoint
    probe_temperatures: dict[str, float]
    ridge_model: RidgeModel
    feature_video_duration_sec: float | None = None
    request_timeout_seconds: float = 900.0


def _finalizer_prediction(checkpoint: Mapping[str, Any]) -> str:
    try:
        return str(checkpoint["finalizer_prediction"])
    except KeyError as exc:
        raise ValueError("development checkpoints require finalizer_prediction") from exc


def _finalizer_cost(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return checkpoint["finalizer_cost"]
    except KeyError as exc:
        raise ValueError("development checkpoints require finalizer_cost") from exc


def reachable_future_cost(
    trajectory: Mapping[str, Any], start: int, end: int | None, scale: float
) -> float:
    """Future native + intermediate probe cost reachable after one checkpoint."""

    current = model_tokens(trajectory["checkpoints"][start]["decision_sunk_cost"])
    if end is None:
        native = max(0.0, model_tokens(trajectory["native_cost"]) - current)
        future = trajectory["checkpoints"][start + 1 : -1]
    else:
        native = max(
            0.0,
            model_tokens(trajectory["checkpoints"][end]["decision_sunk_cost"])
            - current,
        )
        future = trajectory["checkpoints"][start + 1 : end + 1]
    probes = sum(model_tokens(checkpoint["probe_cost"]) for checkpoint in future)
    return (native + probes) / scale


def decision_gap(
    trajectory: Mapping[str, Any], index: int, scale: float, cost_weight: float
) -> float:
    """Stop-now loss minus the best realized later loss on the native trajectory."""

    gold = str(trajectory["gold"])
    checkpoints = trajectory["checkpoints"]
    last = len(checkpoints) - 1
    current = float(_finalizer_prediction(checkpoints[index]) != gold)
    current += cost_weight * model_tokens(_finalizer_cost(checkpoints[index])) / scale

    future = [
        float(_finalizer_prediction(checkpoints[end]) != gold)
        + cost_weight
        * (
            reachable_future_cost(trajectory, index, end, scale)
            + model_tokens(_finalizer_cost(checkpoints[end])) / scale
        )
        for end in range(index + 1, last)
    ]
    future.append(
        float(str(trajectory["native_prediction"]) != gold)
        + cost_weight * reachable_future_cost(trajectory, index, None, scale)
    )
    return current - min(future)


def fit_gap_model(
    trajectories: list[dict[str, Any]], cost_weight: float, alpha: float = 10.0
) -> RidgeModel:
    """Fit one CASE Gap-Ridge model from labeled development trajectories."""

    learnable = [row for row in trajectories if len(row["checkpoints"]) > 1]
    if not learnable:
        raise ValueError("at least one decision trajectory is required")
    scale = statistics.fmean(model_tokens(row["native_cost"]) for row in learnable)

    features: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for row in learnable:
        transitions = len(row["checkpoints"]) - 1
        for index in range(transitions):
            features.append(feature_vector(row, index))
            targets.append(decision_gap(row, index, scale, cost_weight))
            weights.append(1.0 / transitions)
    return RidgeModel(alpha).fit(
        np.asarray(features), np.asarray(targets), np.asarray(weights)
    )
