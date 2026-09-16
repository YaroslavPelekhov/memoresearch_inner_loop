from typing import List
import warnings

import numpy as np
from composer.core import State

from composer.optim.scheduler import ComposerScheduler, _convert_time


def _get_stacked_schedule_lambda(current_step: int, schedule_rows: list):
    first_lr = 0.0
    first_step = 0
    for schedule in schedule_rows:
        if current_step < schedule[0]:
            break
        first_step = schedule[0]
        first_lr = schedule[1]
    else:
        return schedule_rows[-1][1] if len(schedule_rows) else 0.0
    step = (current_step - first_step) / (schedule[0] - first_step)
    if schedule[2] == "linear":
        return first_lr + (schedule[1] - first_lr) * step
    elif schedule[2] == "cosine":
        return first_lr + (1 - np.cos(step * np.pi)) / 2 * (schedule[1] - first_lr)
    else:
        raise ValueError(f"Wrong schedule type [{schedule[2]}]")


def get_actual_schedule_rows(schedule_rows: List, num_training_steps: int):
    """
    Validate and calculate schedule rows.

    Args:
        schedule_rows ('iterable'):
            List of rows, each of these rows describe one LR schedule with last step (fisrt step
            is from previous schedule last step), target LR (LR starts from previous schedule
            target LR), schedule type is 'linear' or 'cosine' changing to schedule target LR.
        num_training_steps (`int`):
            The total number of training steps.
    """
    actual_schedule_rows = []
    first_step = 0
    for last_step, target_lr, sch_type in schedule_rows:
        if -1.0 < last_step <= 1.0:
            last_step = int(last_step * num_training_steps)
        if last_step < 0:
            last_step += num_training_steps
        if last_step <= first_step:
            raise ValueError(
                f"Wrong LR schedule [{schedule_rows}]\nbegin step {first_step}, end step {last_step}"
            )
        if last_step > num_training_steps:
            warnings.warn(
                f"WARNING! Schedule border step is outside training borders "
                f"[{last_step}, {num_training_steps}]"
            )
        actual_schedule_rows.append((last_step, target_lr, sch_type))
        first_step = last_step
    return actual_schedule_rows


class StackedScheduler(ComposerScheduler):
    r"""Decays the learning rate discretely at fixed intervals using selected funcsions.

    Args:
        schedule_rows ('list'):
            List of rows each of which describes one schedule (till step, target LR, changing type).
            List of rows, each of these rows describe one LR schedule with last step (fisrt step
            is from previous schedule last step), target LR (LR starts from previous schedule
            target LR), schedule type is 'linear' or 'cosine' changing to schedule target LR.
    """

    def __init__(
        self,
        schedule_rows: List,
    ):
        self.init_schedule_rows = schedule_rows
        self.schedule_rows = None

    def __call__(self, state: State):
        t_max = _convert_time("1dur", state)
        if self.schedule_rows is None:
            self.schedule_rows = get_actual_schedule_rows(
                self.init_schedule_rows, t_max
            )

        current_time = state.timestamp.get(t_max.unit).value + 1
        lr = _get_stacked_schedule_lambda(
            current_step=current_time, schedule_rows=self.schedule_rows
        )
        return lr
