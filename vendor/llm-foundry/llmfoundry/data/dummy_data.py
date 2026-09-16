# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import numpy as np
import torch
from streaming.base.util import number_abbrev_to_int
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase


class NullTokenizer:
    def __init__(self, bos_token_id: int, eos_token_id: int):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = None


def _get_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 1:
        raise ValueError(f"Distributed world size must be positive. Received {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Distributed rank must satisfy 0 <= rank < world_size. Received rank={rank}, world_size={world_size}."
        )
    return rank, world_size


def _get_shard_bounds(total_size: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    per_shard = (total_size + num_shards - 1) // num_shards
    shard_start = shard_id * per_shard
    shard_end = min(shard_start + per_shard, total_size)
    return shard_start, shard_end


class DummyDataset(IterableDataset):
    """Dummy dataset with random data for debug."""

    seed: int = 42

    def __init__(
        self,
        max_seq_len: int,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        epoch_size: Optional[str | int] = None,
        dataset_size: Optional[int] = None,
        vocab_size: int = 128256,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()

        if epoch_size and dataset_size:
            raise ValueError(
                "arguments `epoch_size` and `epoch_size` "
                "cannot be passed simultaneously"
            )

        size_value = int(1e24)
        if epoch_size:
            size_value = number_abbrev_to_int(epoch_size)
        if dataset_size:
            size_value = dataset_size
        if size_value < 0:
            raise ValueError(f"dataset size cannot be negative. Received {size_value}.")

        self.max_seq_len = max_seq_len
        tokenizer_bos_id = getattr(tokenizer, "bos_token_id", None) if tokenizer else None
        tokenizer_eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
        self._bos_id = bos_token_id if tokenizer_bos_id is None else tokenizer_bos_id
        self._eos_id = eos_token_id if tokenizer_eos_id is None else tokenizer_eos_id
        tokenizer_size = None
        if tokenizer is not None:
            try:
                tokenizer_size = len(tokenizer)
            except TypeError:
                tokenizer_size = getattr(tokenizer, "vocab_size", None)
        self._vocab_size = vocab_size if tokenizer_size is None else tokenizer_size
        self.tokenizer = tokenizer if tokenizer else NullTokenizer(bos_token_id, eos_token_id)
        tokenizer_special_ids = set()
        if tokenizer is not None:
            tokenizer_special_ids.update(
                int(token_id)
                for token_id in getattr(tokenizer, "all_special_ids", [])
                if token_id is not None
            )
        tokenizer_special_ids.update({
            int(self._bos_id),
            int(self._eos_id),
        })
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            tokenizer_special_ids.add(int(pad_token_id))
        self._special_token_ids = sorted(tokenizer_special_ids)
        if min(self._special_token_ids) < 0:
            raise ValueError("Special token ids must be non-negative.")
        if max(self._special_token_ids) >= self._vocab_size:
            raise ValueError(
                "Special token ids must be smaller than the dummy dataset vocabulary size."
            )
        if len(self._special_token_ids) >= self._vocab_size:
            raise ValueError(
                "Dummy dataset vocabulary size must exceed the number of special tokens."
            )
        if self.max_seq_len < 2:
            raise ValueError(
                "Dummy dataset requires max_seq_len >= 2 to place BOS and EOS tokens."
            )
        self._random_token_upper_bound = self._vocab_size - len(self._special_token_ids)
        if self._random_token_upper_bound <= 0:
            raise ValueError(
                "Dummy dataset must have at least one non-special token to sample from."
            )
        self._dataset_size = size_value

    def __iter__(self):
        global_rank, world_size = _get_rank_and_world_size()
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        consumer_id = global_rank * num_workers + worker_id
        num_consumers = world_size * num_workers
        shard_start, shard_end = _get_shard_bounds(
            self._dataset_size,
            consumer_id,
            num_consumers,
        )

        rng = np.random.default_rng(seed=self.seed + consumer_id)

        for _ in range(shard_start, shard_end):
            sample = rng.integers(
                0,
                self._random_token_upper_bound,
                size=self.max_seq_len - 2,
            )
            for special_token_id in self._special_token_ids:
                sample = sample + (sample >= special_token_id)
            sample = np.concatenate([[self._bos_id], sample, [self._eos_id]])
            yield torch.from_numpy(sample)
