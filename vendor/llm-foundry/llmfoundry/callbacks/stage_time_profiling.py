# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Dict

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import TimerType, callback_timer


class StageTimeProfiler(Callback):
    """Callback для профилирования времени, затраченного на каждый этап обучения.

    Этот callback измеряет и логирует время, затраченное на различные стадии
    обучения, включая прямой и обратный проходы, шаги оптимизатора и т.д.
    """

    def __init__(self):
        self.current_batch_timings = {}
        self.epoch_timings = {}
        self.total_timings = {}
        self.start_times = {}
        self.current_batch = 0
        self.current_epoch = 0

    def _start_timer(self, stage_name: str) -> None:
        """Запускает таймер для конкретной стадии."""
        self.start_times[stage_name] = time.time()

    def _stop_timer(self, stage_name: str) -> float:
        """Останавливает таймер для стадии и возвращает затраченное время."""
        if stage_name in self.start_times:
            elapsed_time = time.time() - self.start_times.pop(stage_name)

            # Обновляем текущие метрики батча
            if stage_name not in self.current_batch_timings:
                self.current_batch_timings[stage_name] = 0.0
            self.current_batch_timings[stage_name] += elapsed_time

            # Обновляем метрики эпохи
            if stage_name not in self.epoch_timings:
                self.epoch_timings[stage_name] = 0.0
            self.epoch_timings[stage_name] += elapsed_time

            # Обновляем общие метрики
            if stage_name not in self.total_timings:
                self.total_timings[stage_name] = 0.0
            self.total_timings[stage_name] += elapsed_time

            return elapsed_time
        return 0.0

    def _log_timing(self, state: State, logger: Logger, level: str, stage_name: str, elapsed_time: float) -> None:
        """Логирует время выполнения для конкретной стадии и уровня."""
        logger.log_metrics({f'time_seconds/{level}/{stage_name}': elapsed_time}, step=state.timestamp.batch)

    def _log_timings(self, state: State, logger: Logger, level: str, timings: Dict[str, float]) -> None:
        """Логирует время выполнения для указанного уровня."""
        for stage_name, elapsed_time in timings.items():
            self._log_timing(state, logger, level, stage_name, elapsed_time)

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_start(self, state: State, logger: Logger) -> None:
        """Вызывается перед началом обработки батча."""
        self.current_batch = state.timestamp.batch
        self._start_timer('batch_total')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger) -> None:
        """Вызывается после завершения обработки батча."""
        self._stop_timer('batch_total')

        # Логируем все метрики батча
        self._log_timings(state, logger, 'batch', self.current_batch_timings)

        # Сбрасываем счетчики для следующего батча
        self.current_batch_timings = {}

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def before_forward(self, state: State, logger: Logger) -> None:
        """Вызывается перед прямым проходом."""
        self._start_timer('forward')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def after_forward(self, state: State, logger: Logger) -> None:
        """Вызывается после прямого прохода."""
        self._stop_timer('forward')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def before_loss(self, state: State, logger: Logger) -> None:
        """Вызывается перед вычислением функции потерь."""
        self._start_timer('loss_calculation')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def after_loss(self, state: State, logger: Logger) -> None:
        """Вызывается после вычисления функции потерь."""
        self._stop_timer('loss_calculation')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def before_backward(self, state: State, logger: Logger) -> None:
        """Вызывается перед обратным проходом."""
        self._start_timer('backward')

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def after_backward(self, state: State, logger: Logger) -> None:
        """Вызывается после обратного прохода."""
        self._stop_timer('backward')

    def before_step(self, state: State, logger: Logger) -> None:
        """Вызывается перед шагом оптимизатора."""
        self._start_timer('optimizer_step')

    def after_step(self, state: State, logger: Logger) -> None:
        """Вызывается после шага оптимизатора."""
        self._stop_timer('optimizer_step')

    def before_dataloader(self, state: State, logger: Logger) -> None:
        """Вызывается перед началом работы даталоадера."""
        self._start_timer('dataloader_fetch')

    def after_dataloader(self, state: State, logger: Logger) -> None:
        """Вызывается после завершения работы даталоадера."""
        # Останавливаем таймер операции выборки данных
        elapsed_fetch_time = self._stop_timer('dataloader_fetch')
        self._log_timing(state, logger, 'batch', 'dataloader_fetch', elapsed_fetch_time)

    def eval_start(self, state: State, logger: Logger) -> None:
        """Вызывается перед сохранением чекпойнта."""
        self._start_timer('eval')

    def eval_end(self, state: State, logger: Logger) -> None:
        """Вызывается после сохранения чекпойнта."""
        elapsed_time = self._stop_timer('eval')
        self._log_timing(state, logger, 'eval', 'metrics_calc', elapsed_time)
