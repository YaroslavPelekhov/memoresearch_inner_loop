from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Any, List

import torch
from composer.utils import dist
from composer.core import Callback, State
from composer.loggers import Logger

log = logging.getLogger(__name__)


@dataclass
class DivergenceReport:
    ts: float
    reason: str
    step: int
    loss: float
    ema_loss: float
    threshold: float
    consecutive_steps: int
    ema_beta: float
    offender_rank: Optional[int] = None


class DivergenceDetector(Callback):
    """
    Divergence detector for distributed training.

    Triggers on:
        1) nan/inf loss on ANY rank (checked in after_train_batch)
        2) EMA(global_loss) > threshold for consecutive_steps (checked in after_train_batch)
    """

    def __init__(
        self,
        loss_threshold: float,
        consecutive_steps: int = 100,
        ema_beta: float = 0.98,
        warmup_steps: int = 0,
        dump_path: str = "FAIL_REASON.json",
        print_marker: str = "DIVERGENCE_DETECTED",
        action_on_fail: str = "raise",
    ):
        assert 0.0 < ema_beta < 1.0, "ema_beta must be in (0, 1)"

        self.loss_threshold = float(loss_threshold)
        self.consecutive_steps = int(consecutive_steps)
        self.ema_beta = float(ema_beta)
        self.warmup_steps = int(warmup_steps)
        self.dump_path = str(dump_path)
        self.print_marker = str(print_marker)

        supported_actions = {"raise"}  # TODO: добавить тг-алерты
        assert action_on_fail in supported_actions, (
            f"action_on_fail must be one of {supported_actions}, got '{action_on_fail}'"
        )
        self.action_on_fail = action_on_fail

        self._microbatch_losses: List[torch.Tensor] = []

        # EMA state (maintained on all ranks for consistency)
        self._ema: Optional[float] = None
        self._above_cnt: int = 0

    def _get_device(self, state: State) -> torch.device:
        """Get the current device."""
        if hasattr(state, "device") and hasattr(state.device, "_device"):
            return state.device._device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def after_loss(self, state: State, logger: Logger) -> None:
        loss_t = state.loss
        if loss_t is None:
            return

        if isinstance(loss_t, dict):
            if "total" not in loss_t:
                log.warning(
                    f"[divergence_detector] state.loss is dict without 'total' key: "
                    f"{list(loss_t.keys())}"
                )
                return
            loss_t = loss_t["total"]

        if isinstance(loss_t, torch.Tensor):
            self._microbatch_losses.append(loss_t.detach().clone())
        else:
            device = self._get_device(state)
            self._microbatch_losses.append(
                torch.tensor(float(loss_t), device=device, dtype=torch.float32)
            )

    def after_train_batch(self, state: State, logger: Logger) -> None:
        if not self._microbatch_losses:
            return

        step = int(state.timestamp.batch)
        device = self._microbatch_losses[0].device
        stacked = torch.stack([loss.float().mean() for loss in self._microbatch_losses])
        local_loss = stacked.mean()

        # Clear accumulator
        self._microbatch_losses.clear()

        # Check nan/inf
        local_nonfinite = not torch.isfinite(local_loss).item()
        any_nonfinite, offender_rank = self._any_rank_flag(int(local_nonfinite), device)

        if any_nonfinite:
            self._trigger_fail(
                state,
                logger,
                reason=f"loss_nonfinite_rank{offender_rank}",
                step=step,
                loss=float("nan"),
                ema_loss=self._ema if self._ema is not None else float("nan"),
                offender_rank=offender_rank,
            )

        # Global mean
        batch_loss = self._all_reduce_mean(local_loss, device).item()

        # Update EMA
        if self._ema is None:
            self._ema = batch_loss
        else:
            self._ema = self.ema_beta * self._ema + (1.0 - self.ema_beta) * batch_loss

        # Check threshold (only after warmup)
        trigger = False
        if step >= self.warmup_steps:
            if self._ema > self.loss_threshold:
                self._above_cnt += 1
            else:
                self._above_cnt = 0

            if self._above_cnt >= self.consecutive_steps:
                trigger = True

        # Log metrics
        if dist.get_global_rank() == 0:
            logger.log_metrics(
                {
                    "divergence/ema": self._ema,
                    "divergence/above_count": self._above_cnt,
                    "divergence/threshold": self.loss_threshold,
                    "divergence/batch_loss": batch_loss,
                }
            )

        # Trigger failure if threshold exceeded
        if trigger:
            self._trigger_fail(
                state,
                logger,
                reason=f"ema_loss_above_threshold_{self.consecutive_steps}_steps",
                step=step,
                loss=batch_loss,
                ema_loss=self._ema,
                offender_rank=None,
            )

    def _all_reduce_mean(
        self, tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if dist.get_world_size() == 1:
            return tensor

        result = tensor.clone()
        dist.all_reduce(result, reduce_operation="SUM")
        result = result / dist.get_world_size()
        return result

    def _any_rank_flag(
        self, local_flag: int, device: torch.device
    ) -> tuple[bool, Optional[int]]:
        if dist.get_world_size() == 1:
            return bool(local_flag), (0 if local_flag else None)

        # Check if any rank has the flag
        flag_tensor = torch.tensor([local_flag], dtype=torch.int32, device=device)
        dist.all_reduce(flag_tensor, reduce_operation="MAX")
        any_flag = bool(flag_tensor.item())

        # Find which rank
        offender_rank = None
        if any_flag:
            rank_tensor = torch.tensor(
                [dist.get_global_rank() if local_flag else -1],
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(rank_tensor, reduce_operation="MAX")
            offender_rank = int(rank_tensor.item()) if rank_tensor.item() >= 0 else None

        return any_flag, offender_rank

    def _trigger_fail(
        self,
        state: State,
        logger: Logger,
        *,
        reason: str,
        step: int,
        loss: float,
        ema_loss: float,
        offender_rank: Optional[int],
    ) -> None:
        # Log final metrics
        if dist.get_global_rank() == 0:
            logger.log_metrics(
                {
                    "divergence/detected": 1.0,
                    "divergence/final_loss": loss if math.isfinite(loss) else 0.0,
                    "divergence/final_ema": ema_loss
                    if math.isfinite(ema_loss)
                    else 0.0,
                    "divergence/fail_step": step,
                }
            )

        # Build message
        msg = (
            f"{self.print_marker}: {reason} | "
            f"step={step} | loss={loss:.6g} | ema={ema_loss:.6g} | "
            f"threshold={self.loss_threshold}"
        )
        if offender_rank is not None:
            msg += f" | offender_rank={offender_rank}"

        # Write report file (rank 0 only)
        if dist.get_global_rank() == 0:
            report = DivergenceReport(
                ts=time.time(),
                reason=reason,
                step=step,
                loss=float(loss) if math.isfinite(loss) else float("nan"),
                ema_loss=float(ema_loss) if math.isfinite(ema_loss) else float("nan"),
                threshold=self.loss_threshold,
                consecutive_steps=self.consecutive_steps,
                ema_beta=self.ema_beta,
                offender_rank=offender_rank,
            )
            try:
                dump_dir = os.path.dirname(self.dump_path)
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                with open(self.dump_path, "w") as f:
                    json.dump(asdict(report), f, indent=2)
            except Exception as e:
                log.warning(f"Could not write divergence report: {e}")

        log.error(msg)

        if self.action_on_fail == "raise":
            raise RuntimeError(msg)

    def state_dict(self) -> dict[str, Any]:
        return {
            "_ema": self._ema,
            "_above_cnt": self._above_cnt,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._ema = state_dict.get("_ema")
        self._above_cnt = state_dict.get("_above_cnt", 0)
        self._microbatch_losses = []
