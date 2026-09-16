import os
import pickle
import shutil
from typing import Union

import torch

from composer import Callback, State, Logger, Time, TimeUnit
from composer.utils import dist, format_name_with_dist, format_name_with_dist_and_time, TimerType, callback_timer


def _is_valid_rank():
    return dist.get_global_rank() in (0, 1) and torch.cuda.is_available()


def _validate_interval(interval) -> Time:
    # Check that the interval timestring is parsable and convert into time object
    if isinstance(interval, int):
        interval_ = Time(interval, TimeUnit.BATCH)
    elif isinstance(interval, str):
        interval_ = Time.from_timestring(interval)
    elif isinstance(interval, Time):
        interval_ = interval
    else:
        raise NotImplementedError()
    return interval_


class MemoryStatsSaver(Callback):

    def __init__(
            self,
            folder: str = '{run_name}/memory_stats',
            filename: str = 'ep{epoch}-ba{batch}-rank{rank}.json',
            oom_pickle_name: str = 'rank{rank}-oom.pickle',
            save_interval: Union[int, str, Time] = '2ba',
            collect_memory_snapshot: bool = False
    ) -> None:
        self.save_interval = _validate_interval(save_interval)

        # Verify that the interval has supported units
        if self.save_interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.save_interval.unit}')

        self.folder = folder
        self.filename = filename
        self.oom_pickle_name = oom_pickle_name
        self.last_train_time_value_logged = -1
        self.collect_memory_snapshot = collect_memory_snapshot and torch.cuda.is_available()

        if self.collect_memory_snapshot:
            # print("init memory recording..")
            torch.cuda.memory._record_memory_history(
                enabled="all", max_entries=100000,
            )

    def init(self, state: State, logger: Logger) -> None:

        if self.collect_memory_snapshot:
            oom_filename = os.path.join(
                format_name_with_dist(self.folder, run_name=state.run_name),
                format_name_with_dist(self.oom_pickle_name, run_name=state.run_name),
            )
            shutil.rmtree(oom_filename, ignore_errors=True)

            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print('saving allocated state during OOM')
                snapshot = torch.cuda.memory._snapshot()
                pickle.dump(snapshot, open(oom_filename, 'wb'))

            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

            folder = format_name_with_dist(self.folder, run_name=state.run_name)
            os.makedirs(folder, exist_ok=True)

    def _save_stats(self, state: State):
        if not _is_valid_rank():
            return
        current_time_value = state.timestamp.get(self.save_interval.unit).value
        if current_time_value % self.save_interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            folder = format_name_with_dist(self.folder, run_name=state.run_name)

            timestamp = state.timestamp
            filename = os.path.join(
                folder,
                format_name_with_dist_and_time(self.filename, state.run_name, timestamp),
            )

            if self.collect_memory_snapshot and _is_valid_rank():
                snapshot_filename = filename.replace('.json', '_snapshot.pickle')
                snapshot = torch.cuda.memory._snapshot()

                with open(snapshot_filename, 'wb') as f:
                    pickle.dump(snapshot, f)

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger):
        self._save_stats(state)
