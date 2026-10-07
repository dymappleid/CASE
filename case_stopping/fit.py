"""Calibration, model fitting, and operating-point selection for CASE."""

from __future__ import annotations

from copy import deepcopy
import math
import statistics
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from .core import RidgeModel, fit_gap_model

DEFAULT_COST_WEIGHTS = (0.05, 0.1, 0.2, 0.4, 0.8)
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_BOUNDARIES = tuple(-0.2 + 0.005 * index for index in range(81))
STRICT_SAVING_FLOOR = 0.55
STRICT_ACCURACY_LCB_FLOOR = -0.01
NEAR_OPTIMAL_ACCURACY_PP = 2.0
FALLBACK_SAVING_TARGET = 0.40
_EPSILON = 1e-12


def temperature_scaled_distribution(
    logits: Mapping[str, float], temperature: float
) -> dict[str, float]:
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("logits and a finite positive temperature are required")
    scaled = {str(key): float(value) / temperature for key, value in logits.items()}
    maximum = max(scaled.values())
    masses = {key: math.exp(value - maximum) for key, value in scaled.items()}
    total = sum(masses.values())
    return {key: value / total for key, value in masses.items()}


def fit_temperature(
    examples: Iterable[tuple[Mapping[str, float], str]],
    sample_weights: Iterable[float] | None = None,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> float:
    """Fit one scalar temperature by weighted negative log likelihood."""

    rows = [(dict(logits), str(gold)) for logits, gold in examples]
    if not rows:
        raise ValueError("temperature fitting requires examples")
    weights = [1.0] * len(rows) if sample_weights is None else list(sample_weights)
    if len(weights) != len(rows) or math.fsum(weights) <= 0:
        raise ValueError("invalid temperature-fit weights")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return math.fsum(
            weight
            * -math.log(
                max(
                    temperature_scaled_distribution(logits, temperature)[gold],
                    _EPSILON,
                )
            )
            for (logits, gold), weight in zip(rows, weights)
        ) / math.fsum(weights)

    left, right = math.log(minimum), math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    middle_left = right - ratio * (right - left)
    middle_right = left + ratio * (right - left)
    value_left, value_right = objective(middle_left), objective(middle_right)
    for _ in range(96):
        if value_left <= value_right:
            right, middle_right, value_right = middle_right, middle_left, value_left
            middle_left = right - ratio * (right - left)
            value_left = objective(middle_left)
        else:
            left, middle_left, value_left = middle_left, middle_right, value_right
            middle_right = left + ratio * (right - left)
            value_right = objective(middle_right)
    return math.exp((left + right) / 2.0)


def checkpoint_group(index: int) -> str:
    return str(index) if index < 3 else "3+"


def fit_probe_temperatures(
    trajectories: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Fit the four checkpoint-depth temperatures from training labels only."""

    grouped: dict[str, list[tuple[Mapping[str, float], str]]] = {
        "0": [],
        "1": [],
        "2": [],
        "3+": [],
    }
    for trajectory in trajectories:
        gold = trajectory.get("gold")
        if gold is None:
            raise ValueError("probe calibration requires training labels")
        for index, checkpoint in enumerate(trajectory["checkpoints"]):
            logits = checkpoint.get("probe_logits")
            if logits is not None:
                grouped[checkpoint_group(index)].append((logits, str(gold)))
    return {
        group: fit_temperature(examples) if examples else 1.0
        for group, examples in grouped.items()
    }


def calibrate_trajectories(
    trajectories: Iterable[Mapping[str, Any]], temperatures: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Return copies with calibrated option distributions attached."""

    output: list[dict[str, Any]] = []
    for source in trajectories:
        trajectory = deepcopy(dict(source))
        for index, checkpoint in enumerate(trajectory["checkpoints"]):
            if "probe_logits" in checkpoint:
                checkpoint["calibrated_distribution"] = temperature_scaled_distribution(
                    checkpoint["probe_logits"],
                    float(temperatures[checkpoint_group(index)]),
                )
        output.append(trajectory)
    return output


def fit_models(
    trajectories: list[dict[str, Any]],
    cost_weights: Iterable[float] = DEFAULT_COST_WEIGHTS,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> dict[str, RidgeModel]:
    return {
        str(weight): fit_gap_model(trajectories, float(weight), alpha)
        for weight in cost_weights
    }


def paired_accuracy_lower_bound(
    samples: list[Mapping[str, Any]], alpha: float = 0.05
) -> float:
    """One-sided video-clustered normal lower bound used during calibration."""

    by_video: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        by_video.setdefault(str(sample["video_id"]), []).append(sample)
    differences = [
        statistics.fmean(
            float(row["policy_correct"]) - float(row["native_correct"])
            for row in group
        )
        for group in by_video.values()
    ]
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return mean - NormalDist().inv_cdf(1.0 - alpha) * standard_error


def select_operating_point(
    records: list[dict[str, Any]],
    *,
    saving_floor: float = STRICT_SAVING_FLOOR,
    lcb_floor: float = STRICT_ACCURACY_LCB_FLOOR,
    near_optimal_accuracy_pp: float = NEAR_OPTIMAL_ACCURACY_PP,
    fallback_saving_target: float = FALLBACK_SAVING_TARGET,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Apply the fixed strict/fallback CASE calibration rule."""

    indexed = list(enumerate(records))
    strict = [
        item
        for item in indexed
        if float(item[1]["model_token_saving"]) >= saving_floor - 1e-12
        and float(item[1]["accuracy_lcb"]) >= lcb_floor - 1e-12
    ]
    if strict:
        best_accuracy = max(float(item[1]["accuracy_delta"]) for item in strict)
        tolerance = near_optimal_accuracy_pp / 100.0
        near = [
            item
            for item in strict
            if float(item[1]["accuracy_delta"]) >= best_accuracy - tolerance - 1e-12
        ]
        chosen = max(
            near,
            key=lambda item: (
                float(item[1]["model_token_saving"]),
                float(item[1]["accuracy_delta"]),
                float(item[1]["accuracy_lcb"]),
                item[0],
            ),
        )
        audit = {
            "branch": "strict",
            "candidate_count": len(records),
            "strict_candidate_count": len(strict),
            "near_optimal_candidate_count": len(near),
        }
    else:
        chosen = min(
            indexed,
            key=lambda item: (
                abs(float(item[1]["model_token_saving"]) - fallback_saving_target),
                -float(item[1]["accuracy_lcb"]),
                -float(item[1]["accuracy_delta"]),
                item[0],
            ),
        )
        audit = {
            "branch": "fallback",
            "candidate_count": len(records),
            "strict_candidate_count": 0,
        }
    return chosen[0], chosen[1], audit
