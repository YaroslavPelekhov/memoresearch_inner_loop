# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import tempfile
from typing import Union
from pathlib import Path

import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.utils import (dist, format_name_with_dist_and_time, parse_uri,
                            reproducibility, TimerType, callback_timer)


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


class MonolithicCheckpointSaver(Callback):
    """Save a monolithic checkpoint every N batches.

    Args:
        save_folder (str): Folder to save checkpoints to (can be a URI)
        filename (str): Filename to save checkpoints to.
        batch_interval (int): Number of batches between checkpoints.
        overwrite (bool): Whether to overwrite previous checkpoints.
        keep_optimizer(bool): Whether to save the optimizer state in the monolithic checkpoint.
    """

    def __init__(self,
                 save_folder: str,
                 batch_interval: Union[int, str, Time],
                 filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
                 overwrite: bool = False):
        self.backend, self.bucket_name, self.save_dir_format_str = parse_uri(
            save_folder)
        self.filename_format_str = filename
        self.batch_interval = _validate_interval(batch_interval)
        if self.batch_interval.unit not in [TimeUnit.BATCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.save_interval.unit}')
        self.upload_to_object_store = (self.backend != '')
        self.overwrite = overwrite
        if self.upload_to_object_store:
            self.remote_ud = RemoteUploaderDownloader(
                bucket_uri=f'{self.backend}://{self.bucket_name}')
        else:
            self.remote_ud = None

    def init(self, state: State, logger: Logger):
        if self.upload_to_object_store and self.remote_ud is not None:
            self.remote_ud.init(state, logger)
            # updated_logger_destinations = [*logger.destinations, new_remote_ud]
            # logger.destinations = tuple(updated_logger_destinations)
            state.callbacks.append(self.remote_ud)

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_checkpoint(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_interval.value == 0:
            self._save_checkpoint(state, logger)

    def fit_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_interval.value != 0:
            self._save_checkpoint(state, logger)

    def _save_checkpoint(self, state: State, logger: Logger):
        filename = format_name_with_dist_and_time(self.filename_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        save_dir = format_name_with_dist_and_time(self.save_dir_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        dir_context_mgr = tempfile.TemporaryDirectory(
        ) if self.upload_to_object_store else contextlib.nullcontext(
            enter_result=save_dir)
        with dir_context_mgr as temp_save_dir:
            save_path = str(
                Path(temp_save_dir) /  # type: ignore
                Path(filename))
            dirname = os.path.dirname(save_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            state_dict = {
                'state': state.state_dict(state_dict_type='full'),
                'rng': reproducibility.get_rng_state()
            }

            if dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0:
                torch.save(state_dict, save_path)

            if self.upload_to_object_store and self.remote_ud is not None and \
                (dist.get_global_rank() == 0 or dist.get_fsdp_group_rank() == 0):
                remote_file_name = str(Path(save_dir) / Path(filename))
                self.remote_ud.upload_file(state=state,
                                           remote_file_name=remote_file_name,
                                           file_path=Path(save_path),
                                           overwrite=self.overwrite)
