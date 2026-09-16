from __future__ import annotations

import argparse
import itertools
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROBLEM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROBLEM_DIR))

TARGET_FITNESS = 0.0365
FEEDBACK_FILE = "feedback.json"


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_for_failure(exc: BaseException) -> dict[str, float]:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return {
        "fitness": -1000.0,
        "is_valid": 0.0,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(
        0.5
        * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
    )


def _min_triangle_angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    a = float(np.linalg.norm(p2 - p3))
    b = float(np.linalg.norm(p1 - p3))
    c = float(np.linalg.norm(p1 - p2))
    if min(a, b, c) <= 0.0:
        return 0.0
    cosines = np.clip(
        [
            (b * b + c * c - a * a) / (2 * b * c),
            (a * a + c * c - b * b) / (2 * a * c),
            (a * a + b * b - c * c) / (2 * a * b),
        ],
        -1.0,
        1.0,
    )
    return float(np.degrees(np.min(np.arccos(cosines))))


def _barycentric(points: np.ndarray) -> np.ndarray:
    from helper import get_unit_triangle

    a, b, c = get_unit_triangle()
    v0 = c - a
    v1 = b - a
    v2 = points - a

    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = np.einsum("ij,j->i", v2, v0)
    d21 = np.einsum("ij,j->i", v2, v1)
    denom = d00 * d11 - d01 * d01

    c_weight = (d11 * d20 - d01 * d21) / denom
    b_weight = (d00 * d21 - d01 * d20) / denom
    a_weight = 1.0 - b_weight - c_weight
    return np.column_stack([a_weight, b_weight, c_weight])


def _point_rows(points: np.ndarray) -> list[dict[str, Any]]:
    bary = _barycentric(points)
    rows: list[dict[str, Any]] = []
    for idx, (point, weights) in enumerate(zip(points, bary)):
        rows.append(
            {
                "index": idx,
                "xy": [float(point[0]), float(point[1])],
                "barycentric_abc": [float(x) for x in weights],
                "boundary_slack": float(np.min(weights)),
            }
        )
    return rows


def _triplet_rows(points: np.ndarray, *, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for triplet in itertools.combinations(range(len(points)), 3):
        p1, p2, p3 = (points[i] for i in triplet)
        rows.append(
            {
                "indices": list(triplet),
                "area": _triangle_area(p1, p2, p3),
                "min_angle_deg": _min_triangle_angle_deg(p1, p2, p3),
            }
        )
    rows.sort(key=lambda item: (item["area"], item["min_angle_deg"]))
    return rows[:limit]


def _closest_pair_rows(points: np.ndarray, *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(points)), 2):
        rows.append(
            {
                "indices": [i, j],
                "distance": float(np.linalg.norm(points[i] - points[j])),
            }
        )
    rows.sort(key=lambda item: item["distance"])
    return rows[:limit]


def _point_pressure(worst_triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    area_sum: dict[int, float] = {}
    for row in worst_triplets:
        area = float(row["area"])
        for idx in row["indices"]:
            counts[idx] = counts.get(idx, 0) + 1
            area_sum[idx] = area_sum.get(idx, 0.0) + area
    pressure = [
        {
            "index": idx,
            "worst_triplet_count": count,
            "mean_worst_triplet_area": area_sum[idx] / count,
        }
        for idx, count in counts.items()
    ]
    pressure.sort(
        key=lambda item: (-item["worst_triplet_count"], item["mean_worst_triplet_area"])
    )
    return pressure


def _build_structured_feedback(
    *,
    coordinates: np.ndarray | None,
    metrics: dict[str, float],
    error: str | None,
) -> dict[str, Any]:
    fitness = float(metrics.get("fitness", -1000.0))
    summary = {
        "fitness": fitness,
        "target_fitness": TARGET_FITNESS,
        "target_gap": TARGET_FITNESS - fitness,
        "is_valid": float(metrics.get("is_valid", 0.0)),
        "status": "failed" if error else "valid",
        "error": error,
    }
    feedback: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "heilbron",
        "summary": summary,
        "metrics": metrics,
        "artifacts": [
            {
                "name": FEEDBACK_FILE,
                "path": FEEDBACK_FILE,
                "kind": "geometry_feedback",
                "role": "mutation_context",
                "description": "Full Heilbron geometry feedback for this candidate.",
            }
        ],
    }
    if coordinates is None:
        feedback["diagnosis"] = {
            "primary_issue": "candidate did not produce valid coordinates",
            "mutation_focus": [
                "fix entrypoint execution and output shape before optimizing geometry"
            ],
        }
        return feedback

    points = np.asarray(coordinates, dtype=float)
    if points.shape != (11, 2) or not np.all(np.isfinite(points)):
        feedback.update(
            {
                "output_shape": list(points.shape),
                "diagnosis": {
                    "primary_issue": "candidate output is not a finite (11, 2) coordinate array",
                    "mutation_focus": [
                        "fix entrypoint output shape and finite numeric coordinates before optimizing geometry"
                    ],
                },
            }
        )
        return feedback

    worst_triplets = _triplet_rows(points)
    closest_pairs = _closest_pair_rows(points)
    feedback.update(
        {
            "points": _point_rows(points),
            "worst_triplets": worst_triplets,
            "closest_pairs": closest_pairs,
            "point_pressure": _point_pressure(worst_triplets[:10]),
            "diagnosis": {
                "primary_issue": _primary_issue(metrics),
                "mutation_focus": _mutation_focus(metrics),
                "notes": [
                    "Worst triplets are sorted by triangle area ascending.",
                    "boundary_slack is min barycentric coordinate; small values are close to an edge.",
                    "Use this feedback to make a concise algorithmic change; do not run broad local searches inside mutation.",
                ],
            },
        }
    )
    return feedback


def _primary_issue(metrics: dict[str, float]) -> str:
    if float(metrics.get("is_valid", 0.0)) < 1.0:
        return "invalid candidate"
    if float(metrics.get("degenerate_triangle_count", 0.0)) > 0:
        return "near-degenerate low-area triplets"
    if float(metrics.get("convex_hull_area", 0.0)) < 0.75:
        return "underused enclosing triangle area"
    if float(metrics.get("fitness", 0.0)) < TARGET_FITNESS:
        return "valid layout below target; active bottleneck triplets limit fitness"
    return "target reached"


def _mutation_focus(metrics: dict[str, float]) -> list[str]:
    focus: list[str] = []
    if float(metrics.get("degenerate_triangle_count", 0.0)) > 0:
        focus.append(
            "separate points in the listed worst triplets to remove near-collinearity"
        )
    if float(metrics.get("convex_hull_area", 0.0)) < 0.75:
        focus.append("increase hull usage while preserving inside-triangle constraints")
    if float(metrics.get("spread_y_var", 0.0)) < 0.10:
        focus.append("increase vertical spread; current layout is too flat")
    if float(metrics.get("pairwise_distance_min", 0.0)) < 0.20:
        focus.append("increase spacing for closest point pairs")
    if not focus:
        focus.append(
            "target the repeated point indices in worst_triplets with a small bounded layout rule change"
        )
    return focus


def _write_feedback(candidate_repo: Path, feedback: dict[str, Any]) -> None:
    path = candidate_repo / FEEDBACK_FILE
    path.write_text(json.dumps(_json_safe(feedback), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a repo-harness Heilbron candidate."
    )
    parser.add_argument(
        "--candidate-repo",
        required=True,
        type=Path,
        help="Path to a candidate repo containing solution.py.",
    )
    parser.add_argument(
        "--solution-file",
        default="solution.py",
        help="Candidate solution file relative to --candidate-repo.",
    )
    args = parser.parse_args()

    candidate_repo = args.candidate_repo.resolve()
    candidate_repo_text = str(candidate_repo)
    if candidate_repo_text not in sys.path:
        sys.path.insert(1, candidate_repo_text)

    solution_path = candidate_repo / args.solution_file
    coordinates: np.ndarray | None = None
    error: str | None = None
    try:
        from validate import validate

        module = _load_module(solution_path)
        entrypoint = getattr(module, "entrypoint")
        coordinates = np.asarray(entrypoint(), dtype=float)
        metrics = validate(coordinates)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        metrics = _metrics_for_failure(exc)

    feedback = _build_structured_feedback(
        coordinates=coordinates,
        metrics=metrics,
        error=error,
    )
    _write_feedback(candidate_repo, feedback)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
