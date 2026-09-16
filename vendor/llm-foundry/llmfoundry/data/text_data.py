# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import logging
import inspect
import os
import struct
from collections import defaultdict
from itertools import islice
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase
from llmfoundry.data.data import ENCODING2DTYPE_MAPPING_DICT
from llmfoundry.data.dummy_data import DummyDataset
from llmfoundry.data.fim import FIMTransformer, wrapper_get_item

log = logging.getLogger(__name__)


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        batching_method (str): Which batching method to use, either random, stratified, per_stream,
            or device_per_stream. Defaults to random.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[int] = None,
        predownload: int = 100_000,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = "orig",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1b",
        shuffle_seed: int = 9176,
        shuffle_block_size: int = 1 << 18,
        sampling_method: str = "balanced",
        batching_method: str = "random",
        sft_mode: bool = False,
        legacy_batching: bool = True,
        include_profiler_info: bool = False,
        **kwargs: Any,
    ):

        group_method = kwargs.pop("group_method", None)
        if group_method is not None:
            raise NotImplementedError(
                "group_method is deprecated and has been removed.\nTo "
                + "concatenate, use the --concat_tokens "
                + "argument when creating your MDS dataset with concat_c4.py"
            )

        self.split_to_fimrate = kwargs.pop(
            "split_to_fimrate", defaultdict(lambda: defaultdict(float))
        )

        for split_name, fim_rate in list(self.split_to_fimrate.items()):
            if isinstance(fim_rate, (int, float)):
                self.split_to_fimrate[split_name] = {"vanilla_fim": float(fim_rate)}

        self.use_syntax_aware_fim = kwargs.pop("use_syntax_aware_fim", False)

        if len(kwargs) > 0:
            raise ValueError(
                f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}"
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f"local directory {local} does not contain split {split}"
                    )

        # Build Dataset
        streaming_kwargs = dict(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            batching_method=batching_method,
        )
        if "legacy" in inspect.signature(StreamingDataset.__init__).parameters:
            streaming_kwargs["legacy"] = legacy_batching
        super().__init__(**streaming_kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sft_mode = sft_mode

        # Note (Sbr): this wrapper adds 'split' field to sample
        # and allows you to add custom logic for tokens of different datasets
        for shard in self.shards:
            shard.get_item = wrapper_get_item(shard.get_item).__get__(
                shard, shard.__class__
            )

        self.fim_transformer = FIMTransformer(
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            fim_prefix_token="<|fim_prefix|>",
            fim_middle_token="<|fim_middle|>",
            fim_suffix_token="<|fim_suffix|>",
        )

        self.include_profiler_info = include_profiler_info

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample: Mapping):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                "If tokenizing on-the-fly, tokenizer must have a pad_token_id"
            )
        return self.tokenizer(
            text_sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
        )

    def _read_binary_tokenized_sample(
        self, sample: Dict[str, Any], field: str = "tokens"
    ):
        # take dtype or use np.int64
        sample_dtype_code = sample["dtype"] if "dtype" in sample else 2
        assert sample_dtype_code in ENCODING2DTYPE_MAPPING_DICT, (
            f"Use unsupported dtype code{sample_dtype_code}"
        )
        tokens_dtype = ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
        return torch.from_numpy(
            np.frombuffer(sample[field], dtype=tokens_dtype)[: self.max_seq_len]
            .copy()
            .astype(np.int64)
        )

    def _read_dpo_binary_tokenized_sample(self, sample: Dict[str, Any]):
        # take dtype or use np.int64
        sample_dtype_code = sample["dtype"] if "dtype" in sample else 0
        assert sample_dtype_code in ENCODING2DTYPE_MAPPING_DICT, (
            f"Use unsupported dtype code{sample_dtype_code}"
        )
        tokens_dtype = ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
        for k in sample.keys():
            if k in ["q_ids", "ab_ids", "aw_ids"]:
                sample[k] = torch.from_numpy(
                    np.array(np.frombuffer(sample[k], dtype=np.uint32).astype(np.int64))
                )
            elif k in ["q_mask", "ab_mask", "aw_mask"]:
                sample[k] = torch.from_numpy(
                    np.array(np.frombuffer(sample[k], dtype=np.int32))
                )
            elif k == "margin":
                sample[k] = struct.unpack("f", sample[k])[: self.max_seq_len]
            elif k == "q_hash":
                sample[k] = sample[k].decode("utf-8")
        return sample

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "tokens" in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        elif "ab_ids" in sample:
            token_sample = self._read_dpo_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                "StreamingTextDataset needs samples to have a `text` or `tokens` column"
            )
        sample_fim_rates = self.split_to_fimrate[sample["split"]]
        if sum(sample_fim_rates.values()) > 0:
            assert not self.sft_mode, (
                "Using FIM in combination with SFT Mode is not supported yet."
            )
            token_sample = self.fim_transformer.transform(
                token_sample, fim_rates=sample_fim_rates
            )

        if self.sft_mode:
            labels_sample = self._read_binary_tokenized_sample(sample, "labels")
            return token_sample, labels_sample

        if self.include_profiler_info:
            sample_split = sample.get("split", "")
            if isinstance(token_sample, torch.Tensor):
                return {"input_ids": token_sample, "_split": sample_split}
            token_sample["_split"] = sample_split
        return token_sample


class SplitTrackingCollatorWrapper:
    """Strips '_split' metadata from dataset items, calls the real collator,
    then attaches the collected splits to the batch as '_splits'."""

    def __init__(self, base_collator: Callable):
        self.base_collator = base_collator

    def __call__(self, examples: List[Any]) -> Dict[str, Any]:
        splits = None
        if examples and isinstance(examples[0], dict) and "_split" in examples[0]:
            splits = [ex.pop("_split") for ex in examples]
            # Restore the original format downstream collators expect
            if all(len(ex) == 1 and "input_ids" in ex for ex in examples):
                examples = [ex["input_ids"] for ex in examples]

        batch = self.base_collator(examples)

        if splits is not None and isinstance(batch, dict):
            batch["_splits"] = splits
        return batch


class DpoCollatorWrapper:
    # collator to train on data with preferences

    def __init__(self, max_token_len, pad_token_id, pad_to_max_len=False):
        self.pad_batch_margins = True
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id if pad_token_id is not None else 2
        self.pad_to_max_len = pad_to_max_len
        tp_sp_group_size = dist.get_tp_sp_group_size()
        self.divider = tp_sp_group_size if tp_sp_group_size is not None else 8

        self.divider *= 2
        print(
            f"[INFO]: Doubled divider in DpoCollatorWrapper (from {self.divider // 2} to {self.divider} to make up for splitting the seq in two halves in the end"
        )

    def __call__(self, data: List[Any]) -> Dict[str, torch.Tensor]:
        """
        every entry in `data` has the following keys:
        q - the question asked
        ab - a good answer to the question
        aw - a bad answer to the question
        margin - subjective difference between the answers
        len(data) equals to the global batch size, but every entry should be
        decomposed to two strings

        data is converted to the list [ab0, ab1, ..., abn, aw0, aw1, ..., awn]
        """
        # data = [torch.load('/home/jovyan/jserdyuk/sample/onebatch.data')]
        inputs = [
            np.concatenate((s["q_ids"], s["ab_ids"]), dtype=np.int64) for s in data
        ] + [np.concatenate((s["q_ids"], s["aw_ids"]), dtype=np.int64) for s in data]
        inputs = [
            s[: self.max_token_len] if len(s) > self.max_token_len else s
            for s in inputs
        ]
        inputs = [torch.tensor(s, dtype=torch.long) for s in inputs]
        inputs = pad_sequence(inputs, padding_value=self.pad_token_id, batch_first=True)

        act_mask = [
            np.concatenate((s["q_mask"], s["ab_mask"]), dtype=np.int64) for s in data
        ] + [np.concatenate((s["q_mask"], s["aw_mask"]), dtype=np.int64) for s in data]
        act_mask = [
            s[: self.max_token_len] if len(s) > self.max_token_len else s
            for s in act_mask
        ]
        act_mask = [torch.tensor(s, dtype=torch.long) for s in act_mask]
        act_mask = pad_sequence(act_mask, padding_value=0, batch_first=True)

        margins = torch.tensor(
            [s.get("margin", 0.0) for s in data], dtype=torch.float32
        )

        if inputs.size(1) % self.divider != 0:
            pad_length = self.divider - inputs.size(1) % self.divider
            if pad_length > 0:
                inputs = torch.nn.functional.pad(
                    inputs,
                    pad=(0, pad_length),
                    mode="constant",
                    value=self.pad_token_id,
                )
                act_mask = torch.nn.functional.pad(
                    act_mask, pad=(0, pad_length), mode="constant", value=0
                )

        if self.pad_to_max_len:
            pad_length = self.max_token_len - inputs.size(1)
            if pad_length > 0:
                inputs = torch.nn.functional.pad(
                    inputs,
                    pad=(0, pad_length),
                    mode="constant",
                    value=self.pad_token_id,
                )
                act_mask = torch.nn.functional.pad(
                    act_mask, pad=(0, pad_length), mode="constant", value=0
                )

        # now the size of input_ids is 2*batch_size, so this list is reshaped from
        # [ab0, ab1, aw0, aw1] to [[ab0, aw0], [ab1, aw1]]
        # the final shape is [batch_size, 2, length]
        # reshape converts [ab0, ab1, aw0, aw1] to [[ab0, ab1], [aw0, aw1]]
        # transpose(0, 1) turns [[ab0, ab1], [aw0, aw1]] into [[ab0, aw0], [ab1, aw1]], so the
        # corresponding answers are combined

        return {
            "input_ids": inputs.reshape(
                2, inputs.shape[0] // 2, inputs.shape[1]
            ).transpose(0, 1),
            "attention_mask": inputs.not_equal(self.pad_token_id)
            .long()
            .reshape(2, inputs.shape[0] // 2, inputs.shape[1])
            .transpose(0, 1),
            "action_mask": act_mask.reshape(
                2, inputs.shape[0] // 2, inputs.shape[1]
            ).transpose(0, 1),
            "margins": margins,
        }


class FlashAttnCollatorWrapper:
    def __init__(
        self,
        base_collator: Callable,
        eos_token: Optional[int],
        bos_token: Optional[int],
        pad_token: Optional[int] = None,
        boundary_mode: str = "auto",  # "auto" | "eos" | "bos" | "eos_bos"
    ):
        self.base_collator = base_collator
        assert eos_token is not None or bos_token is not None, \
            "FlashAttnCollatorWrapper requires bos or eos token specified"
        self.eos_token_id = eos_token
        self.bos_token_id = bos_token
        self.pad_token_id = pad_token
        self.boundary_mode = boundary_mode

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        assert "input_ids" in batch and "labels" in batch, \
            "FlashAttnCollatorWrapper expects at least 'input_ids' and 'labels' in batch"

        batch["position_ids"] = self.calculate_position_ids(batch["input_ids"], pad_token_id=self.pad_token_id)
        return batch

    def calculate_position_ids(self, input_ids: torch.Tensor, pad_token_id: Optional[int] = None) -> torch.Tensor:
        bs, seq_len = input_ids.shape

        mode = self.boundary_mode
        if mode == "auto":
            mode = "eos" if self.eos_token_id is not None else "bos"

        if mode == "eos_bos":
            assert self.eos_token_id is not None and self.bos_token_id is not None, \
                "boundary_mode='eos_bos' requires both eos_token_id and bos_token_id"
            return self._calculate_position_ids_eos_bos(input_ids, pad_token_id)

        if mode == "eos":
            assert self.eos_token_id is not None
            return self._calculate_position_ids_eos(input_ids, pad_token_id)

        if mode == "bos":
            assert self.bos_token_id is not None
            return self._calculate_position_ids_bos(input_ids, pad_token_id)

        device = input_ids.device
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)

    def _calculate_position_ids_bos(self, input_ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
        bs, seqlen = input_ids.shape
        device = input_ids.device

        idx = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)

        not_pad = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        if pad_token_id is not None:
            not_pad = input_ids.ne(pad_token_id)

        starts = input_ids.eq(self.bos_token_id) & not_pad
        starts[:, 0] = not_pad[:, 0]

        start_idx = torch.where(starts, idx, torch.full_like(idx, -10**9))
        last_start = start_idx.cummax(dim=1).values
        return idx - last_start

    def _calculate_position_ids_eos(self, input_ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
        bs, seqlen = input_ids.shape
        device = input_ids.device

        idx = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)

        not_pad = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        if pad_token_id is not None:
            not_pad = input_ids.ne(pad_token_id)

        prev_is_eos = torch.cat(
            [
                torch.ones((bs, 1), dtype=torch.bool, device=device),
                input_ids[:, :-1].eq(self.eos_token_id),
            ],
            dim=1,
        )

        starts = prev_is_eos & not_pad
        starts[:, 0] = not_pad[:, 0]

        start_idx = torch.where(starts, idx, torch.full_like(idx, -10**9))
        last_start = start_idx.cummax(dim=1).values
        return idx - last_start

    def _calculate_position_ids_eos_bos(self, input_ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
        bs, seqlen = input_ids.shape
        device = input_ids.device

        idx = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)

        not_pad = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        if pad_token_id is not None:
            not_pad = input_ids.ne(pad_token_id)

        is_bos = input_ids.eq(self.bos_token_id)
        prev_is_eos = torch.cat(
            [
                torch.ones((bs, 1), dtype=torch.bool, device=device),
                input_ids[:, :-1].eq(self.eos_token_id),
            ],
            dim=1,
        )

        # BOS after EOS -> start; plus force start at first token
        starts = is_bos & prev_is_eos & not_pad
        starts[:, 0] = not_pad[:, 0]

        start_idx = torch.where(starts, idx, torch.full_like(idx, -10**9))
        last_start = start_idx.cummax(dim=1).values
        return idx - last_start


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator: Callable,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
    ):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                "Must supply a value for either eos_token_id or bos_token_id, but got None for both."
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. "
                + "Please supply `eos_token_id` if sequences end with an EOS token, or use "
                + "`bos_token_id` if sequences start with a BOS token."
            )

        self.split_token_id = eos_token_id
        self.bos_mode = False
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)  # type: ignore
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


class PaddingCollatorWrapper:
    def __init__(self, pad_token: int, max_seq_len: int, scale: float = 1.0):
        """Pad samples so they are divisible by TP*SP size. Apply additional scaling
        to divider if provided.
        """
        self.pad_token_id = pad_token
        self.max_seq_len = max_seq_len
        tp_sp_group_size = dist.get_tp_sp_group_size()

        self.divider = tp_sp_group_size if tp_sp_group_size is not None else 1
        assert self.divider * scale == int(self.divider * scale), (
            "Devider is expected to be an int value but provided scale makes it non-int. "
            f"Scale={scale}, divider={self.divider}, divider * scale={self.divider * scale}."
        )
        self.divider = int(self.divider * scale)

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        for input_ids, labels in examples:
            assert torch.any(labels != -100), (
                "All labels equal -100. Probably, max_seq_len is too small."
            )

        batch = {"input_ids": [], "labels": []}

        local_max_seq_len = 0
        for input_ids, labels in examples:
            assert len(input_ids) == len(labels)
            assert len(input_ids) <= self.max_seq_len
            local_max_seq_len = max(local_max_seq_len, len(input_ids))

        log.debug(
            f"__COLLATE_INFO__ max len = {local_max_seq_len}; batch size = {len(examples)}"
        )

        if local_max_seq_len % self.divider != 0:
            local_max_seq_len = self.divider * (local_max_seq_len // self.divider + 1)

        assert local_max_seq_len <= self.max_seq_len
        for input_ids, labels in examples:
            pad_size = local_max_seq_len - len(input_ids)

            assert pad_size >= 0
            batch["input_ids"].append(
                torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=self.pad_token_id
                )
            )
            batch["labels"].append(
                torch.nn.functional.pad(labels, (0, pad_size), value=-100)
            )

        for k in batch.keys():
            batch[k] = torch.vstack(batch[k])
        batch["local_max_seq_len"] = local_max_seq_len
        return batch


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Optional[PreTrainedTokenizerBase],
    device_batch_size: int,
):
    assert cfg.name == "text", (
        f"Tried to build text dataloader with cfg.name={cfg.name}"
    )
    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset_hf.py"
        )

    # get kwargs
    streams_dict = cfg.dataset.pop("streams", None)
    mlm_probability = cfg.dataset.pop("mlm_probability", None)
    eos_token_id = cfg.get("eos_token_id", None)
    bos_token_id = cfg.get("bos_token_id", None)
    use_dummy = cfg.dataset.pop("use_dummy", None)  # adding dummy flag
    enable_profiler = cfg.get("enable_data_profiler", True)

    sft_mode = cfg.get("sft_mode", False)
    # Public Streaming requires the per-device DataLoader batch size for
    # deterministic partitioning; the omitted private fork inferred it.
    additional_dataset_kwargs = {"batch_size": device_batch_size}
    if not use_dummy:
        global_fim_rates = cfg.dataset.pop("fim_rates", {})
        if not global_fim_rates:
            legacy_fim_rate = cfg.dataset.pop("fim_rate", 0.0)
            if legacy_fim_rate > 0:
                global_fim_rates = {"vanilla_fim": legacy_fim_rate}
            else:
                global_fim_rates = {
                    "vanilla_fim": 0.0,
                    "inline_fim": 0.0,
                    "multiline_fim": 0.0,
                    "syntax_aware_fim": 0.0,
                }
        split_to_fimrate = defaultdict(lambda: defaultdict(float))
        for fim_type, rate in global_fim_rates.items():
            split_to_fimrate["default"][fim_type] = rate
        # build streams
        streams = None
        if streams_dict is not None:
            streams = []
            for _, stream_kw in streams_dict.items():
                stream_fim_rates = stream_kw.pop("fim_rates", {})
                legacy_stream_fim_rate = stream_kw.pop("fim_rate", None)
                if legacy_stream_fim_rate is not None and not stream_fim_rates:
                    stream_fim_rates = {"vanilla_fim": legacy_stream_fim_rate}
                if stream_fim_rates:
                    for fim_type, rate in stream_fim_rates.items():
                        split_to_fimrate[stream_kw["split"]][fim_type] = rate
                else:
                    for fim_type, rate in global_fim_rates.items():
                        split_to_fimrate[stream_kw["split"]][fim_type] = rate
                streams.append(Stream(**stream_kw))
        for split_name in split_to_fimrate:
            if split_name != "default" and not split_to_fimrate[split_name]:
                split_to_fimrate[split_name] = split_to_fimrate["default"].copy()
        if "default" in split_to_fimrate:
            del split_to_fimrate["default"]
        converted_split_to_fimrate = {
            outer_key: dict(inner_dd)
            for outer_key, inner_dd in split_to_fimrate.items()
        }
        cfg.dataset.split_to_fimrate = converted_split_to_fimrate
        # build dataset potentially with streams
        # ---
        # HARDCODE: set cache_limit to 512bg to prevent tmp from overflowing
        # this prevents OOM error for jobs and cloud workers from hanging
        dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            streams=streams,
            cache_limit="256gb",
            **cfg.dataset,
            sft_mode=sft_mode,
            include_profiler_info=enable_profiler,
            **additional_dataset_kwargs,
        )
    else:
        # build dataset with dummy data
        dataset = DummyDataset(
            tokenizer=tokenizer, **cfg.dataset, **additional_dataset_kwargs
        )

    if not sft_mode:
        collate_fn = transformers.DataCollatorForLanguageModeling(
            tokenizer=dataset.tokenizer,
            mlm=mlm_probability is not None,
            mlm_probability=mlm_probability,
        )
    else:
        # Training sft model
        if hasattr(cfg, "pad_token_id") and cfg.pad_token_id is not None:
            pad_token_id = cfg.pad_token_id
        else:
            pad_token_id = 0

        padding_scale = cfg.get("padding_scale", 1.0)
        collate_fn = PaddingCollatorWrapper(
            pad_token=pad_token_id,
            max_seq_len=cfg.dataset.max_seq_len,
            scale=padding_scale,
        )
    if cfg.get("batch_type", None) == "flash_attn":
        pad_token_for_flash = pad_token_id if sft_mode else None
        if sft_mode:
            if eos_token_id is None or bos_token_id is None:
                raise ValueError(
                    "SFT + flash_attn requires both boundary tokens: eos_token_id and bos_token_id. "
                    "Sequence boundaries are detected using the [EOS][BOS] pattern."
                )
            boundary_mode = "eos_bos"
        else:
            boundary_mode = "eos" if eos_token_id is not None else "bos"
        collate_fn = FlashAttnCollatorWrapper(
            base_collator=collate_fn,
            eos_token=eos_token_id,
            bos_token=bos_token_id,
            pad_token=pad_token_for_flash,
            boundary_mode=boundary_mode,
        )

    if cfg.get("dpo_data", None):
        # Training sft model
        if hasattr(cfg, "pad_token_id") and cfg.pad_token_id is not None:
            pad_token_id = cfg.pad_token_id
        else:
            pad_token_id = 0

        collate_fn = DpoCollatorWrapper(
            max_token_len=cfg.get("max_seq_len", None),
            pad_token_id=pad_token_id,
            pad_to_max_len=cfg.get("pad_to_max_len", False),
        )

    additional_dataloader_kwargs = {}
    if sft_mode:
        additional_dataloader_kwargs["shuffle"] = False

    if enable_profiler:
        collate_fn = SplitTrackingCollatorWrapper(collate_fn)

    num_workers = cfg.num_workers
    prefetch_factor = cfg.get("prefetch_factor", 2)
    persistent_workers = cfg.get("persistent_workers", True)
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        timeout=cfg.get("timeout", 0),
        **additional_dataloader_kwargs,
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == "__main__":
    import argparse

    from llmfoundry.utils.builders import build_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help="the name of the tokenizer to use",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="the path to the local copy of the dataset",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default=None,
        help="the path to the remote copy to stream from (optional)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="which split of the dataset to use"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=32, help="max sequence length to test"
    )

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}"
        )
    else:
        print(f"Reading {args.split} split from {args.local_path}")

    cfg = {
        "name": "text",
        "dataset": {
            "local": args.local_path,
            "remote": args.remote_path,
            "split": args.split,
            "shuffle": False,
            "max_seq_len": args.max_seq_len,
            "keep_zip": True,  # in case we need compressed files after testing
        },
        "drop_last": False,
        "num_workers": 4,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    tokenizer_cfg = {"name": args.tokenizer, "kwargs": {}}
    tokenizer_cfg["kwargs"] = {"model_max_length": args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print("\n")
        print("#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch["input_ids"]):
            print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
            print(tokenizer.decode(token_sample))
