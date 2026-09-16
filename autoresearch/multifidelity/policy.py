"""Uncertainty-aware shrinking-gate decision policy."""

from __future__ import annotations

from autoresearch.multifidelity.models import (
    Decision,
    FidelityRung,
    GateDecision,
    ProbabilityEstimate,
)


class ShrinkingGatePolicy:
    """Use interval bounds, not point estimates, for irreversible decisions."""

    def __init__(self, *, observe_only: bool) -> None:
        self.observe_only = observe_only

    def decide(
        self,
        estimate: ProbabilityEstimate | None,
        rung: FidelityRung,
        *,
        final_rung: bool = False,
    ) -> GateDecision:
        if final_rung:
            recommended = Decision.PROMOTE
            reason = "full budget completed"
        elif estimate is None:
            recommended = Decision.MORE_EVIDENCE
            reason = "no calibrated probability estimate is available"
        elif estimate.upper < rung.kill_gate:
            recommended = Decision.KILL
            reason = (
                f"upper probability bound {estimate.upper:.6f} is below "
                f"kill gate {rung.kill_gate:.6f}"
            )
        elif estimate.lower > rung.promote_gate:
            recommended = Decision.PROMOTE
            reason = (
                f"lower probability bound {estimate.lower:.6f} is above "
                f"promote gate {rung.promote_gate:.6f}"
            )
        else:
            recommended = Decision.MORE_EVIDENCE
            reason = "probability interval intersects the uncertainty region"

        executed = recommended
        if self.observe_only and not final_rung:
            executed = Decision.MORE_EVIDENCE
            reason += "; observe-only mode continues the trajectory"
        return GateDecision(
            recommended=recommended,
            executed=executed,
            reason=reason,
            kill_gate=rung.kill_gate,
            promote_gate=rung.promote_gate,
            observe_only=self.observe_only,
        )
