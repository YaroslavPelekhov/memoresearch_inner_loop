"""Leakage-resistant fitting for multi-fidelity probability artifacts."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Literal

import numpy as np

from autoresearch.multifidelity.models import ProbeFeatures
from autoresearch.multifidelity.predictor import (
    LinearPredictor,
    LinearRungModel,
    ProbabilityModel,
)

ModelVariant = Literal["trajectory_only", "probe_aware"]

TRAJECTORY_FEATURES = ("main_fitness", "delta_fitness")
PROBE_FEATURES = (
    "main_fitness",
    "delta_fitness",
    "probe_fitness",
    "local_headroom",
    "probe_progress",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    max_iterations: int = 100,
) -> np.ndarray:
    if matrix.ndim != 2 or len(matrix) != len(labels):
        raise ValueError("logistic matrix and labels have incompatible shapes")
    if len(set(float(value) for value in labels)) != 2:
        raise ValueError("logistic fitting requires both outcome classes")
    design = np.column_stack((np.ones(len(matrix)), matrix))
    weights = np.zeros(design.shape[1], dtype=float)
    regularizer = np.eye(design.shape[1], dtype=float) * l2
    regularizer[0, 0] = 1.0e-8
    for _ in range(max_iterations):
        probability = _sigmoid(design @ weights)
        variance = np.clip(probability * (1.0 - probability), 1.0e-8, None)
        gradient = design.T @ (probability - labels) + regularizer @ weights
        hessian = design.T @ (design * variance[:, None]) + regularizer
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1.0e-9:
            break
    return weights


def _feature_names(records: list[dict[str, Any]], variant: ModelVariant) -> list[str]:
    candidates = TRAJECTORY_FEATURES if variant == "trajectory_only" else PROBE_FEATURES
    return [
        name
        for name in candidates
        if any(record.get(name) is not None for record in records)
    ]


def _standardized_matrix(
    records: list[dict[str, Any]],
    feature_names: list[str],
    *,
    defaults: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    if defaults is None:
        defaults = {}
        for name in feature_names:
            present = [
                float(record[name])
                for record in records
                if record.get(name) is not None
            ]
            if not present:
                raise ValueError(f"feature {name!r} has no observed values")
            defaults[name] = float(np.mean(present))
    raw = np.array(
        [
            [
                float(record[name]) if record.get(name) is not None else defaults[name]
                for name in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("model features must be finite")
    if scales is None:
        scales = {
            name: max(float(np.std(raw[:, index])), 1.0e-8)
            for index, name in enumerate(feature_names)
        }
    standardized = np.array(
        [
            [
                (raw[row, column] - defaults[name]) / scales[name]
                for column, name in enumerate(feature_names)
            ]
            for row in range(len(records))
        ],
        dtype=float,
    )
    return standardized, defaults, scales


def _raw_linear_model(
    weights: np.ndarray,
    feature_names: list[str],
    defaults: dict[str, float],
    scales: dict[str, float],
) -> LinearPredictor:
    coefficients = {
        name: float(weights[index + 1] / scales[name])
        for index, name in enumerate(feature_names)
    }
    intercept = float(
        weights[0]
        - sum(
            weights[index + 1] * defaults[name] / scales[name]
            for index, name in enumerate(feature_names)
        )
    )
    return LinearPredictor(
        intercept=intercept,
        coefficients=coefficients,
        feature_defaults=defaults,
    )


def _fit_probability_head(
    train: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    feature_names: list[str],
    *,
    l2: float,
) -> LinearPredictor:
    train_matrix, defaults, scales = _standardized_matrix(train, feature_names)
    train_labels = np.array([float(record["eventual_winner"]) for record in train])
    base_weights = _fit_logistic(train_matrix, train_labels, l2=l2)
    calibration_matrix, _, _ = _standardized_matrix(
        calibration,
        feature_names,
        defaults=defaults,
        scales=scales,
    )
    base_logits = (
        np.column_stack((np.ones(len(calibration)), calibration_matrix)) @ base_weights
    )
    calibration_labels = np.array(
        [float(record["eventual_winner"]) for record in calibration]
    )
    platt = _fit_logistic(base_logits[:, None], calibration_labels, l2=l2)
    slope = max(0.0, float(platt[1]))
    if slope == 0.0:
        prevalence = float(np.mean(calibration_labels))
        calibrated_intercept = math.log(prevalence / (1.0 - prevalence))
        calibrated_weights = np.zeros_like(base_weights)
        calibrated_weights[0] = calibrated_intercept
    else:
        calibrated_weights = base_weights * slope
        calibrated_weights[0] += float(platt[0])
    return _raw_linear_model(
        calibrated_weights,
        feature_names,
        defaults,
        scales,
    )


def _fit_fitness_head(
    train: list[dict[str, Any]],
    feature_names: list[str],
    *,
    l2: float,
) -> LinearPredictor:
    matrix, defaults, scales = _standardized_matrix(train, feature_names)
    design = np.column_stack((np.ones(len(matrix)), matrix))
    targets = np.array([float(record["final_fitness"]) for record in train])
    if not np.all(np.isfinite(targets)):
        raise ValueError("final fitness values must be finite")
    regularizer = np.eye(design.shape[1], dtype=float) * l2
    regularizer[0, 0] = 1.0e-8
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return _raw_linear_model(weights, feature_names, defaults, scales)


def _cluster_resample(
    records: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["group_id"])].append(record)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("cluster bootstrap requires at least two lineage groups")
    sample: list[dict[str, Any]] = []
    for _ in groups:
        sample.extend(grouped[rng.choice(groups)])
    return sample


def _validate_records(records: list[dict[str, Any]], *, expected_split: str) -> None:
    if not records:
        raise ValueError(f"{expected_split} split is empty")
    seen: set[tuple[str, int]] = set()
    for record in records:
        if record.get("split") != expected_split:
            raise ValueError(f"record does not belong to {expected_split}")
        key = (str(record["run_id"]), int(record["budget_batches"]))
        if key in seen:
            raise ValueError(f"duplicate run/budget record: {key}")
        seen.add(key)
        if not isinstance(record.get("eventual_winner"), bool):
            raise ValueError("eventual_winner must be boolean")


def fit_probability_model(
    train_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    *,
    variant: ModelVariant,
    model_id: str,
    split_manifest_sha256: str,
    train_split_sha256: str,
    probability_calibration_split_sha256: str,
    locked_test_split_sha256: str,
    bootstrap_resamples: int = 200,
    confidence_level: float = 0.95,
    seed: int = 41,
    l2: float = 1.0,
) -> ProbabilityModel:
    """Fit per-rung point models and cluster-bootstrap probability intervals."""

    _validate_records(train_records, expected_split="train")
    _validate_records(
        calibration_records,
        expected_split="probability_calibration",
    )
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be nonnegative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    train_by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    calibration_by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        train_by_budget[int(record["budget_batches"])].append(record)
    for record in calibration_records:
        calibration_by_budget[int(record["budget_batches"])].append(record)
    if set(train_by_budget) != set(calibration_by_budget):
        raise ValueError("train and probability-calibration budgets differ")

    rng = random.Random(seed)
    probability_models: dict[int, LinearRungModel] = {}
    fitness_models: dict[int, LinearPredictor] = {}
    for budget in sorted(train_by_budget):
        train = train_by_budget[budget]
        calibration = calibration_by_budget[budget]
        feature_names = _feature_names(train, variant)
        point = _fit_probability_head(train, calibration, feature_names, l2=l2)
        fitness_models[budget] = _fit_fitness_head(
            train,
            feature_names,
            l2=l2,
        )
        bootstrap_models: list[LinearPredictor] = []
        for _ in range(bootstrap_resamples):
            sampled_train = _cluster_resample(train, rng)
            sampled_calibration = _cluster_resample(calibration, rng)
            try:
                bootstrap_models.append(
                    _fit_probability_head(
                        sampled_train,
                        sampled_calibration,
                        feature_names,
                        l2=l2,
                    )
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
        minimum_successful = max(1, math.ceil(bootstrap_resamples / 2))
        if len(bootstrap_models) < minimum_successful:
            raise ValueError(
                f"budget {budget} produced only {len(bootstrap_models)} "
                f"successful bootstrap fits; need {minimum_successful}"
            )
        probability_models[budget] = LinearRungModel(
            **point.model_dump(),
            interval_radius=0.0,
            bootstrap_models=bootstrap_models,
            confidence_level=confidence_level,
            calibration_method="heldout_platt_cluster_bootstrap",
        )

    return ProbabilityModel(
        model_id=model_id,
        calibrated=True,
        frozen=True,
        variant=variant,
        split_manifest_sha256=split_manifest_sha256,
        train_split_sha256=train_split_sha256,
        probability_calibration_split_sha256=(probability_calibration_split_sha256),
        locked_test_split_sha256=locked_test_split_sha256,
        per_budget=probability_models,
        fitness_per_budget=fitness_models,
    )


def features_from_record(record: dict[str, Any]) -> ProbeFeatures:
    return ProbeFeatures.model_validate(
        {name: record.get(name) for name in PROBE_FEATURES}
    )
