"""Feature construction for the trajectory/probe ablation."""

from __future__ import annotations

from autoresearch.multifidelity.models import ProbeFeatures


def fitness_from_loss(loss: float) -> float:
    if loss < 0.0:
        raise ValueError("loss cannot be negative")
    return 1.0 / (1.0 + loss)


def build_probe_features(
    *,
    main_fitness: float,
    previous_main_fitness: float | None,
    probe_fitness: float | None,
    previous_probe_fitness: float | None,
) -> ProbeFeatures:
    """Build F, delta-F, H=V-F, and E=V-V(previous)."""

    return ProbeFeatures(
        main_fitness=main_fitness,
        delta_fitness=(
            None
            if previous_main_fitness is None
            else main_fitness - previous_main_fitness
        ),
        probe_fitness=probe_fitness,
        local_headroom=(
            None if probe_fitness is None else probe_fitness - main_fitness
        ),
        probe_progress=(
            None
            if probe_fitness is None or previous_probe_fitness is None
            else probe_fitness - previous_probe_fitness
        ),
    )
