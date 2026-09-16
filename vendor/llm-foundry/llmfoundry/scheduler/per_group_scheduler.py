from __future__ import annotations

from typing import Dict

from composer.optim.scheduler import ComposerScheduler


class PerGroupComposerScheduler:
    """
    Container for per-optimizer-param-group Composer schedulers.
    Holds a default :class:`~composer.optim.scheduler.ComposerScheduler` and optional
    per-optimizer-param-group overrides.

    Used with :func:`composer.optim.scheduler.compile_composer_scheduler`, which detects
    ``per_group_schedulers`` and builds a :class:`torch.optim.lr_scheduler.LambdaLR` with
    one multiplier function per optimizer param group.

    Args:
        default_scheduler: Scheduler applied to any param group without an override.
        per_group_schedulers: Maps optimizer ``param_groups`` index (0 = default/remainder
            group, 1+ = matched ``param_groups`` from the optimizer config) to a scheduler.
    """

    def __init__(
        self,
        default_scheduler: ComposerScheduler,
        per_group_schedulers: Dict[int, ComposerScheduler],
    ) -> None:
        self.default_scheduler = default_scheduler
        self.per_group_schedulers = dict(per_group_schedulers)

    def get_scheduler_for_group(self, group_idx: int) -> ComposerScheduler:
        """Return the scheduler for ``optimizer.param_groups[group_idx]``."""
        return self.per_group_schedulers.get(group_idx, self.default_scheduler)
