"""Composer-native gradient clipping with diagnostic telemetry."""

from __future__ import annotations

import logging
from typing import Any

from composer.algorithms import GradientClipping
from composer.core import Event, State
from composer.loggers import Logger
import torch
from torch.distributed.fsdp import FullyShardedDataParallel

from autoresearch.gradient_metrics import gradient_norm_metrics

log = logging.getLogger(__name__)


def _nonfinite_gradient_names(state: State) -> list[str]:
    """Return parameter names containing at least one non-finite gradient."""

    return [
        name
        for name, parameter in state.model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]


class LoggedGradientClipping(GradientClipping):
    """Preserve Composer norm clipping while logging its pre-clipping norm.

    PyTorch's norm-clipping functions return the total norm measured before
    scaling. Composer's stock algorithm discards that value; this subclass
    publishes it without performing a second clipping pass.
    """

    def __init__(
        self,
        *,
        clipping_type: str,
        clipping_threshold: float,
        log_interval: int = 1,
    ) -> None:
        if clipping_type != "norm":
            raise ValueError(
                "LoggedGradientClipping supports only clipping_type='norm'"
            )
        if log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        super().__init__(
            clipping_type=clipping_type,
            clipping_threshold=clipping_threshold,
        )
        self.log_interval = log_interval

    def apply(self, event: Event, state: State, logger: Logger) -> int | None:
        if event == Event.INIT:
            return super().apply(event, state, logger)
        if event != Event.AFTER_TRAIN_BATCH or state.deepspeed_enabled:
            return super().apply(event, state, logger)

        total_norm: torch.Tensor | float | None = None
        if state.fsdp_enabled:
            for module in state.model.modules():
                if (
                    isinstance(module, FullyShardedDataParallel)
                    and module.check_is_root()
                ):
                    total_norm = module.clip_grad_norm_(
                        max_norm=self.clipping_threshold
                    )
                    break
        else:
            try:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    state.model.parameters(),
                    max_norm=self.clipping_threshold,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                names = _nonfinite_gradient_names(state)
                batch = state.timestamp.batch.value
                logger.log_metrics(
                    {
                        "gradient_clipping/nonfinite": 1.0,
                        "gradient_clipping/nonfinite_parameter_count": float(
                            len(names)
                        ),
                    }
                )
                rendered_names = ",".join(names) or "<finite-elements-overflow>"
                log.error(
                    "[autoresearch] nonfinite_gradient batch=%s parameters=%s",
                    batch,
                    rendered_names,
                )
                raise RuntimeError(
                    f"Non-finite gradient at batch {batch}: {rendered_names}"
                ) from exc

        if total_norm is None:
            # Preserve Composer's behavior for an unusual FSDP topology where
            # no root module is visible to this rank.
            return super().apply(event, state, logger)
        if state.timestamp.batch.value % self.log_interval == 0:
            norm_value = (
                float(total_norm.detach().item())
                if isinstance(total_norm, torch.Tensor)
                else float(total_norm)
            )
            logger.log_metrics(
                gradient_norm_metrics(norm_value, self.clipping_threshold)
            )
        return None


def build_logged_gradient_clipping(
    *,
    clipping_type: str,
    clipping_threshold: float,
    diagnostics: Any,
) -> LoggedGradientClipping:
    """Build the diagnostic algorithm from an OmegaConf-compatible mapping."""

    log_interval = 1
    if diagnostics is not None:
        log_interval = int(diagnostics.get("log_interval", log_interval))
    return LoggedGradientClipping(
        clipping_type=clipping_type,
        clipping_threshold=clipping_threshold,
        log_interval=log_interval,
    )
