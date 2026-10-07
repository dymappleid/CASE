"""Offline application and calibration-time evaluation of frozen CASE policies."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    FEATURE_NAMES,
    FrozenPolicy,
    OperatingPoint,
    RidgeModel,
    activation_enabled,
    canonical_cost,
    feature_vector,
    model_tokens,
)
from .fit import DEFAULT_BOUNDARIES, paired_accuracy_lower_bound, select_operating_point


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_policy(path: str | Path, host: str, model: str) -> FrozenPolicy:
    """Load one frozen policy from a CASE policy bundle."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "case-frozen-policies-v1":
        raise ValueError("unsupported frozen-policy file")
    if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("frozen policy uses a different feature definition")
    try:
        item = payload["policies"][host][model]
    except KeyError as exc:
        raise KeyError(f"no frozen CASE policy for {host}/{model}") from exc
    return FrozenPolicy(
        host=host,
        model=model,
        operating_point=OperatingPoint(
            float(item["cost_weight"]),
            float(item["boundary"]),
            str(item["activation"]),
        ),
        probe_temperatures={
            str(key): float(value)
            for key, value in item["probe_temperatures"].items()
        },
        ridge_model=RidgeModel.from_dict(item["ridge_model"]),
        feature_video_duration_sec=(
            None
            if item.get("feature_video_duration_sec") is None
            else float(item["feature_video_duration_sec"])
        ),
        request_timeout_seconds=float(item.get("request_timeout_seconds", 900.0)),
    )


def _apply_operating_point(
    trajectories: Iterable[Mapping[str, Any]],
    model: RidgeModel,
    operating_point: OperatingPoint,
) -> list[dict[str, Any]]:
    """Apply one fitted model/operating point without reading labels."""

    decisions: list[dict[str, Any]] = []
    for trajectory in trajectories:
        active = activation_enabled(
            str(trajectory["duration_group"]), operating_point.activation
        )
        state_count = max(0, len(trajectory["checkpoints"]) - 1)
        decision = {
            "sample_id": str(trajectory["sample_id"]),
            "video_id": str(trajectory["video_id"]),
            "activated": active,
            "probe_count": 0,
            "stopped": False,
            "stop_checkpoint_index": None,
            "stop_score": None,
        }
        if active:
            for index in range(state_count):
                score = float(model.predict(feature_vector(trajectory, index)[None, :])[0])
                decision["probe_count"] = index + 1
                if score <= operating_point.boundary:
                    decision.update(
                        stopped=True,
                        stop_checkpoint_index=index,
                        stop_score=score,
                    )
                    break
        decisions.append(decision)
    return decisions


def apply_policy(
    trajectories: Iterable[Mapping[str, Any]], policy: FrozenPolicy
) -> list[dict[str, Any]]:
    """Apply a frozen CASE policy without reading labels or correctness fields."""

    return _apply_operating_point(
        trajectories, policy.ridge_model, policy.operating_point
    )


def _sum_costs(costs: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    total = canonical_cost({})
    for cost in costs:
        value = canonical_cost(cost)
        for key in total:
            total[key] += value[key]
    return total


def _policy_cost(
    trajectory: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, float]:
    if not decision["activated"]:
        return canonical_cost(trajectory["native_cost"])

    probes = [
        trajectory["checkpoints"][index]["probe_cost"]
        for index in range(int(decision["probe_count"]))
    ]
    if not decision["stopped"]:
        return _sum_costs([trajectory["native_cost"], *probes])

    checkpoint = trajectory["checkpoints"][int(decision["stop_checkpoint_index"])]
    if "finalizer_cost" not in checkpoint:
        raise ValueError("offline evaluation of a STOP requires finalizer_cost")
    return _sum_costs(
        [checkpoint["decision_sunk_cost"], *probes, checkpoint["finalizer_cost"]]
    )


def evaluate_decisions(
    trajectories: list[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate a fixed policy on an explicitly labeled development/eval set."""

    by_sample = {str(row["sample_id"]): row for row in trajectories}
    samples: list[dict[str, Any]] = []
    for decision in decisions:
        trajectory = by_sample[str(decision["sample_id"])]
        if "gold" not in trajectory:
            raise ValueError("evaluation requires gold labels")
        if decision["stopped"]:
            checkpoint = trajectory["checkpoints"][
                int(decision["stop_checkpoint_index"])
            ]
            if "finalizer_prediction" not in checkpoint:
                raise ValueError("offline STOP evaluation requires finalizer_prediction")
            prediction = str(checkpoint["finalizer_prediction"])
        else:
            prediction = str(trajectory["native_prediction"])

        native_cost = canonical_cost(trajectory["native_cost"])
        policy_cost = _policy_cost(trajectory, decision)
        gold = str(trajectory["gold"])
        samples.append(
            {
                "sample_id": str(trajectory["sample_id"]),
                "video_id": str(trajectory["video_id"]),
                "native_correct": float(str(trajectory["native_prediction"]) == gold),
                "policy_correct": float(prediction == gold),
                "stopped": bool(decision["stopped"]),
                "native_cost": native_cost,
                "policy_cost": policy_cost,
            }
        )
    return summarize(samples)


def summarize(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    native_costs = [row["native_cost"] for row in samples]
    policy_costs = [row["policy_cost"] for row in samples]
    native_total = _sum_costs(native_costs)
    policy_total = _sum_costs(policy_costs)

    def cost_view(key: str) -> dict[str, float]:
        native = native_total[key]
        policy = policy_total[key]
        return {
            "native_total": native,
            "policy_total": policy,
            "saving": 0.0 if not native else 1.0 - policy / native,
        }

    native_accuracy = sum(float(row["native_correct"]) for row in samples) / len(samples)
    policy_accuracy = sum(float(row["policy_correct"]) for row in samples) / len(samples)
    native_model_tokens = sum(model_tokens(cost) for cost in native_costs)
    policy_model_tokens = sum(model_tokens(cost) for cost in policy_costs)
    return {
        "sample_count": len(samples),
        "video_count": len({str(row["video_id"]) for row in samples}),
        "native_accuracy": native_accuracy,
        "policy_accuracy": policy_accuracy,
        "accuracy_delta": policy_accuracy - native_accuracy,
        "accuracy_lcb": paired_accuracy_lower_bound(samples),
        "stop_rate": sum(bool(row["stopped"]) for row in samples) / len(samples),
        "text_token": cost_view("text_token"),
        "visual_token": cost_view("visual_token"),
        "model_token": {
            "native_total": native_model_tokens,
            "policy_total": policy_model_tokens,
            "saving": 0.0
            if not native_model_tokens
            else 1.0 - policy_model_tokens / native_model_tokens,
        },
        "samples": samples,
    }


def search_operating_points(
    trajectories: list[Mapping[str, Any]],
    models: Mapping[str, RidgeModel],
    boundaries: Iterable[float] = DEFAULT_BOUNDARIES,
) -> dict[str, Any]:
    """Evaluate the fixed grid on calibration data and apply the CASE selector."""

    records: list[dict[str, Any]] = []
    for weight_key, model in sorted(models.items(), key=lambda item: float(item[0])):
        for boundary in boundaries:
            operating_point = OperatingPoint(float(weight_key), float(boundary), "medium_long")
            metrics = evaluate_decisions(
                trajectories, _apply_operating_point(trajectories, model, operating_point)
            )
            records.append(
                {
                    "operating_point": asdict(operating_point),
                    "accuracy_lcb": float(metrics["accuracy_lcb"]),
                    "accuracy_delta": float(metrics["accuracy_delta"]),
                    "model_token_saving": float(metrics["model_token"]["saving"]),
                    "visual_token_saving": float(metrics["visual_token"]["saving"]),
                    "stop_rate": float(metrics["stop_rate"]),
                }
            )
    index, selected, audit = select_operating_point(records)
    return {
        "selected": selected,
        "selected_index": index,
        "selection": audit,
        "candidates": records,
    }
