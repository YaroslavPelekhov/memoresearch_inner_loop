from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
from dataclasses import dataclass, asdict
from typing import Any, Optional

import json
import time
import torch
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s |  %(message)s"
)
log = logging.getLogger(__name__)


@dataclass
class ProcessorFallbackReport:
    ts: float
    reason: str
    step: int
    total_fallbacks: int
    max_processor_fallback_count: Optional[int]


class ProcessorFallbackCounter(Callback):
    """
    Callback that counts processor fallbacks and stops training when limit is exceeded.
    It monitors batches via BEFORE_BATCH_SPLIT_DATALOADER hook to track fallback events.

    Args:
        max_processor_fallback_count: Maximum allowed processor fallbacks
            If None, no limit is set
        dump_path: Path to save JSON report when fallback limit is exceeded
            Can be absolute or relative to the working directory

    Raises:
        RuntimeError: When max_processor_fallback_count is exceeded
    """

    def __init__(
        self,
        max_processor_fallback_count: Optional[int] = 10000,
        dump_path: str = "FAIL_REASON.json",
    ):
        assert (
            max_processor_fallback_count is None or max_processor_fallback_count >= 0
        ), (
            f"max_processor_fallback_count must be >= 0 or None, got {max_processor_fallback_count}"
        )

        assert dump_path is not None, "dump_path must be provided"

        self.max_processor_fallback_count = max_processor_fallback_count
        self.dump_path = dump_path

        self.total_processor_fallbacks: int = 0
        self.modality_fallbacks: dict[str, int] = {}
        self.print_marker = "[ProcessorFallbackCounter]"

    def before_batch_split_dataloader(self, state: State, logger: Logger):
        batch = state.raw_batch

        modality_fallback_keys = [
            key for key in batch.keys() if key.endswith("_processor_fallback_count")
        ]
        local_processor_fallbacks = [batch[key] for key in modality_fallback_keys]

        if dist.get_world_size() > 1:
            processor_fallbacks_tensor = torch.tensor(
                local_processor_fallbacks, dtype=torch.int64, device="cuda"
            )
            dist.all_reduce(processor_fallbacks_tensor, reduce_operation="SUM")
            global_processor_fallbacks = processor_fallbacks_tensor.cpu().tolist()
        else:
            global_processor_fallbacks = local_processor_fallbacks

        modality_metrics = {}
        for key, global_fallback_count in zip(
            modality_fallback_keys, global_processor_fallbacks
        ):
            modality = key.replace("_processor_fallback_count", "")

            if modality not in self.modality_fallbacks:
                self.modality_fallbacks[modality] = 0

            self.modality_fallbacks[modality] += global_fallback_count
            self.total_processor_fallbacks += global_fallback_count

            metric_name = f"trainer/{modality}_processor_fallbacks"
            modality_metrics[metric_name] = self.modality_fallbacks[modality]

        # Log metrics only for rank 0
        if dist.get_global_rank() == 0:
            logger.log_metrics(
                {
                    **modality_metrics,
                    "trainer/total_processor_fallbacks": self.total_processor_fallbacks,
                }
            )

        processor_fallback_count_exceeded = (
            self.max_processor_fallback_count is not None
            and self.total_processor_fallbacks > self.max_processor_fallback_count
        )

        if processor_fallback_count_exceeded:
            self._trigger_fail(
                state,
                logger,
                reason="processor_fallback_count_exceeded_threshold",
                step=int(state.timestamp.batch),
                total_fallbacks=self.total_processor_fallbacks,
                max_processor_fallback_count=self.max_processor_fallback_count,
            )

    def _trigger_fail(
        self,
        state: State,
        logger: Logger,
        *,
        reason: str,
        step: int,
        total_fallbacks: int,
        max_processor_fallback_count: Optional[int],
    ) -> None:
        msg = (
            f"{self.print_marker}: {reason} | "
            f"step={step} | "
            f"total_fallbacks_count={total_fallbacks} | "
            f"max_processor_fallback_count={max_processor_fallback_count}"
        )

        if dist.get_global_rank() == 0:
            report = ProcessorFallbackReport(
                ts=time.time(),
                reason=reason,
                step=step,
                total_fallbacks=total_fallbacks,
                max_processor_fallback_count=max_processor_fallback_count,
            )
            try:
                dump_dir = os.path.dirname(self.dump_path)
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                with open(self.dump_path, "w") as f:
                    json.dump(asdict(report), f, indent=2)

                log.info(f"{self.print_marker} Report saved to {self.dump_path}")

            except Exception as e:
                log.error(
                    f"{self.print_marker} Could not write processor fallback report: {e}",
                    exc_info=True,
                )

        log.error(msg)
        raise RuntimeError(msg)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_processor_fallbacks": self.total_processor_fallbacks,
            "modality_fallbacks": self.modality_fallbacks,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_processor_fallbacks = state_dict.get("total_processor_fallbacks", 0)
        self.modality_fallbacks = state_dict.get("modality_fallbacks", {})

        if dist.get_global_rank() == 0:
            log.info(
                f"{self.print_marker} Restored state: "
                f"total_processor_fallbacks={self.total_processor_fallbacks}, "
                f"modality_fallbacks={self.modality_fallbacks}"
            )
