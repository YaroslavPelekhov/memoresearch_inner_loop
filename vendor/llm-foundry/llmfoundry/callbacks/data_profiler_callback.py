"""Data Profiler callback — tracks per-split loss statistics during training."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as torch_dist
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist

log = logging.getLogger(__name__)

_AGG_COUNT = 0
_AGG_SUM = 1
_AGG_SUM_SQ = 2
_AGG_HIGH = 3
_AGG_LOW = 4
_AGG_FIELDS = 5


class DataProfilerCallback(Callback):
    """Tracks EMA mean / variance of per-sample loss for every data split
    and counts samples with anomalously high or low loss.
    Optionally saves outlier samples to the disk.

    Args:
        log_interval:  How often (in batches) to push metrics to the logger.
        anomaly_threshold:  Number of standard deviations for anomaly detection.
        min_samples_for_anomaly:  Minimum samples per split before counting anomalies.
        ema_alpha:  Smoothing factor for the exponential moving average
            (per sample).  Larger values make the statistics adapt faster.
        outlier_save_path:  If provided, outlier samples are saved as JSONL to
            ``{outlier_save_path}/{split}/high.rank{R}.{N}.jsonl`` and
            ``{outlier_save_path}/{split}/low.rank{R}.{N}.jsonl``.
        max_outlier_file_mb:  Maximum size (in MB) of a single JSONL file before
            rotating to a new one.  Default 100 MB.
    """

    def __init__(
        self,
        log_interval: Union[int, str, Time] = "100ba",
        anomaly_threshold: float = 3.0,
        min_samples_for_anomaly: int = 100,
        ema_alpha: float = 0.001,
        outlier_save_path: Optional[str] = None,
        max_outlier_file_mb: int = 100,
    ):
        if isinstance(log_interval, int):
            self.log_interval = Time(log_interval, TimeUnit.BATCH)
        elif isinstance(log_interval, str):
            self.log_interval = Time.from_timestring(log_interval)
        else:
            self.log_interval = log_interval

        self.anomaly_threshold = anomaly_threshold
        self.min_samples_for_anomaly = min_samples_for_anomaly
        self.ema_alpha = ema_alpha
        self.outlier_save_path = outlier_save_path
        self.max_outlier_file_bytes = max_outlier_file_mb * 1024 * 1024
        self._outlier_buffer: list = []
        self._file_counters: dict = defaultdict(int)
        # EMA stats for anomaly detection (count is Python int, mean/var are
        # GPU scalar tensors lazily initialised on first encounter).
        self.split_stats: dict = {}
        # Interval aggregates for logging: split_name -> GPU tensor [5]
        # (count, loss_sum, loss_sum_sq, high_count, low_count).
        # Accumulated between log intervals, then reduced and cleared.
        self._interval_agg: dict = {}

    @staticmethod
    def _get_model_config(state: State):
        inner = DataProfilerCallback._get_inner_model(state)
        if hasattr(inner, "config"):
            return inner.config
        module = getattr(inner, "module", inner)
        return getattr(module, "config", None)

    @staticmethod
    def _get_per_sample_losses(state: State):
        config = DataProfilerCallback._get_model_config(state)
        if config is None:
            return None
        losses = getattr(config, "_data_profiler_per_sample_losses", None)
        if losses is not None:
            config._data_profiler_per_sample_losses = None
        return losses

    def _get_outlier_filepath(self, split_name: str, outlier_type: str) -> Path:
        save_dir = Path(self.outlier_save_path) / split_name
        save_dir.mkdir(parents=True, exist_ok=True)

        rank = dist.get_global_rank()
        key = (split_name, outlier_type)
        counter = self._file_counters[key]
        while True:
            filepath = save_dir / f"{outlier_type}.rank{rank}.{counter:05d}.jsonl"
            if (
                not filepath.exists()
                or filepath.stat().st_size < self.max_outlier_file_bytes
            ):
                break
            counter += 1
        self._file_counters[key] = counter
        return filepath

    def _flush_outliers(self, records: list):
        by_key: dict = defaultdict(list)
        for record in records:
            by_key[(record["_split"], record["_type"])].append(record)

        for (split_name, outlier_type), group in by_key.items():
            filepath = self._get_outlier_filepath(split_name, outlier_type)
            with open(filepath, "a") as f:
                for record in group:
                    out = {k: v for k, v in record.items() if not k.startswith("_")}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

    @staticmethod
    def _get_inner_model(state: State):
        model = state.model
        if hasattr(model, "module"):
            model = model.module
        return getattr(model, "model", model)

    def _get_or_create_split_stats(self, split_name, device):
        s = self.split_stats.get(split_name)
        if s is None:
            s = {
                "count": 0,
                "mean": torch.tensor(0.0, device=device),
                "var": torch.tensor(0.0, device=device),
            }
            self.split_stats[split_name] = s
        return s

    def _get_or_create_agg(self, split_name, device):
        agg = self._interval_agg.get(split_name)
        if agg is None:
            agg = torch.zeros(_AGG_FIELDS, dtype=torch.float64, device=device)
            self._interval_agg[split_name] = agg
        return agg

    def fit_start(self, state: State, logger: Logger):
        config = self._get_model_config(state)
        tp_size = getattr(config, "tp_size", 1) or 1
        assert tp_size == 1, (
            f"You are attempting to use the data profiler with tp_size={tp_size} > 1. "
            "The statistics aggregation may not work correctly with tensor parallelism. "
            "If you are fine with that, comment this assert and continue. "
            "You may also turn the data profiler off if you don't need it."
        )
        config._data_profiler_enabled = True

    def after_forward(self, state: State, logger: Logger):
        if not state.model.training:
            return

        per_sample_losses = self._get_per_sample_losses(state)
        if per_sample_losses is None:
            return

        batch = state.batch
        splits = batch.get("_splits") if isinstance(batch, dict) else None
        if splits is None:
            splits = ["unknown"] * per_sample_losses.shape[0]

        losses = per_sample_losses.float()
        device = losses.device
        buffer_outliers = self.outlier_save_path is not None

        split_indices: dict = defaultdict(list)
        for i, name in enumerate(splits):
            split_indices[name].append(i)

        for split_name, indices in split_indices.items():
            idx = torch.tensor(indices, device=device)
            split_losses = losses[idx]
            n = len(indices)

            s = self._get_or_create_split_stats(split_name, device)
            agg = self._get_or_create_agg(split_name, device)

            agg[_AGG_COUNT] += n
            agg[_AGG_SUM] += split_losses.sum().double()
            agg[_AGG_SUM_SQ] += split_losses.pow(2).sum().double()

            if s["count"] >= self.min_samples_for_anomaly:
                std = s["var"].sqrt()
                high_mask = split_losses > (s["mean"] + self.anomaly_threshold * std)
                low_mask = split_losses < (s["mean"] - self.anomaly_threshold * std)
                n_high = high_mask.sum()
                n_low = low_mask.sum()
                agg[_AGG_HIGH] += n_high.double()
                agg[_AGG_LOW] += n_low.double()

                if buffer_outliers and (n_high + n_low).item() > 0:
                    mean_val = s["mean"].item()
                    std_val = std.item()
                    for local_i in high_mask.nonzero(as_tuple=False).view(-1):
                        batch_i = indices[local_i]
                        self._buffer_outlier(
                            state,
                            batch,
                            batch_i,
                            split_name,
                            split_losses[local_i].item(),
                            mean_val,
                            std_val,
                            "high",
                        )
                    for local_i in low_mask.nonzero(as_tuple=False).view(-1):
                        batch_i = indices[local_i]
                        self._buffer_outlier(
                            state,
                            batch,
                            batch_i,
                            split_name,
                            split_losses[local_i].item(),
                            mean_val,
                            std_val,
                            "low",
                        )

            # Batch EMA update (pure GPU when no outliers)
            if s["count"] == 0:
                s["mean"] = split_losses.mean()
                if n > 1:
                    s["var"] = split_losses.var()
            else:
                eff_alpha = 1 - (1 - self.ema_alpha) ** n
                batch_mean = split_losses.mean()
                diff = batch_mean - s["mean"]
                old_mean = s["mean"]
                s["mean"] = old_mean + eff_alpha * diff
                batch_mse = (split_losses - old_mean).pow(2).mean()
                s["var"] = (1 - eff_alpha) * s["var"] + eff_alpha * batch_mse
            s["count"] += n

    def _buffer_outlier(
        self, state, batch, sample_idx, split_name, loss_val, mean, std, outlier_type
    ):
        input_ids = (
            batch["input_ids"][sample_idx].tolist()
            if isinstance(batch, dict) and "input_ids" in batch
            else []
        )
        labels = (
            batch["labels"][sample_idx].tolist()
            if isinstance(batch, dict) and "labels" in batch
            else None
        )

        record = {
            "_split": split_name,
            "_type": outlier_type,
            "step": state.timestamp.batch.value,
            "loss": round(loss_val, 6),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "z_score": round((loss_val - mean) / std, 4) if std > 0 else 0.0,
            "input_ids": input_ids,
        }
        if labels is not None:
            record["labels"] = labels
        self._outlier_buffer.append(record)

    def batch_end(self, state: State, logger: Logger):
        step = state.timestamp.batch.value
        if step == 0 or step % self.log_interval.value != 0:
            return

        world_size = dist.get_world_size()
        local_splits = list(self._interval_agg.keys())

        if world_size > 1:
            all_split_sets = [None] * world_size
            torch_dist.all_gather_object(all_split_sets, local_splits)
            all_splits = list(
                dict.fromkeys(s for splits in all_split_sets for s in splits)
            )
        else:
            all_splits = local_splits

        if not all_splits:
            self._outlier_buffer.clear()
            return

        first_agg = next(iter(self._interval_agg.values()))
        device = first_agg.device
        tensor = torch.zeros(
            len(all_splits) * _AGG_FIELDS, dtype=torch.float64, device=device
        )
        for i, split_name in enumerate(all_splits):
            agg = self._interval_agg.get(split_name)
            if agg is not None:
                offset = i * _AGG_FIELDS
                tensor[offset : offset + _AGG_FIELDS] = agg

        if world_size > 1:
            torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.SUM)

        if dist.get_global_rank() == 0:
            metrics = {}
            for i, split_name in enumerate(all_splits):
                offset = i * _AGG_FIELDS
                count = int(tensor[offset + _AGG_COUNT].item())
                if count == 0:
                    continue
                loss_sum = tensor[offset + _AGG_SUM].item()
                loss_sum_sq = tensor[offset + _AGG_SUM_SQ].item()
                high_count = int(tensor[offset + _AGG_HIGH].item())
                low_count = int(tensor[offset + _AGG_LOW].item())

                mean = loss_sum / count
                std = max(0.0, loss_sum_sq / count - mean**2) ** 0.5

                p = f"data_profiler/{split_name}"
                metrics[f"{p}/mean_loss"] = mean
                metrics[f"{p}/std_loss"] = std
                metrics[f"{p}/high_outliers_fraction"] = high_count / count
                metrics[f"{p}/low_outliers_fraction"] = low_count / count
                metrics[f"{p}/count"] = count

            if metrics:
                logger.log_metrics(metrics)

        if self.outlier_save_path is not None and self._outlier_buffer:
            self._flush_outliers(self._outlier_buffer)

        self._outlier_buffer.clear()
        self._interval_agg.clear()
