# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import json
import typing as tp

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from streaming import Stream, StreamingDataset
from composer.core import Evaluator, DataSpec

from llmfoundry.data.processors.gigaspeech_processtor import FEATURE_EXTRACTOR_REGISTRY
from llmfoundry.data.data import ENCODING2DTYPE_MAPPING_DICT


class StreamingAudioDataset(StreamingDataset):
    NUM_RETRIES: int = 10

    def __init__(
        self,
        feature_extractor_cfg: tp.Optional[DictConfig] = None,
        **kwargs: tp.Dict,
    ):
        super().__init__(**kwargs)
        assert feature_extractor_cfg is not None

        feature_extractor_cfg = om.to_container(feature_extractor_cfg)
        feature_extractor_type = feature_extractor_cfg.pop("feature_extractor_type", "audioaug")

        if feature_extractor_type not in FEATURE_EXTRACTOR_REGISTRY:
            raise ValueError(
                f"{feature_extractor_type} is not supported, available types are {FEATURE_EXTRACTOR_REGISTRY.keys()}"
            )

        if feature_extractor_type == "audioaug":
            feature_extractor_cfg: dict = {"extraction_config": feature_extractor_cfg}

        self.featurizer = FEATURE_EXTRACTOR_REGISTRY[feature_extractor_type](**feature_extractor_cfg)

    def _read_binary_tokenized_sample(self, sample: tp.Dict[str, tp.Any], field: str = "tokens") -> torch.Tensor:
        """
        Reads a binary tokenized sample from a given field in the sample dictionary.

        Args:
        sample (Dict[str, Any]): Sample containing the binary encoded field.
        field (str, optional): The key in the sample dict to read. Defaults to "tokens".

        Returns:
        torch.Tensor: The tokenized data as a tensor.
        """
        sample_dtype_code = sample['dtype'] if 'dtype' in sample else 2
        assert sample_dtype_code in ENCODING2DTYPE_MAPPING_DICT, f"Use unsupported dtype code{sample_dtype_code}"
        tokens_dtype = ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
        return torch.from_numpy(
            np.frombuffer(sample[field], dtype=tokens_dtype).copy().astype(np.int64)
        )

    def _get_decoded_str_from_deserialized_bytes(self, deserialized_bytes: torch.Tensor) -> tp.Union[tp.List, str]:
        char_list = [chr(code) for code in deserialized_bytes]
        decoded_strings = "".join(char_list).strip().split("\n")
        if len(decoded_strings) == 1 and decoded_strings[0] == "":
            return []
        return decoded_strings

    def _get_spectrogram_tensor_from_filepath(self, audio_filepath: str) -> torch.Tensor:
        audio_features, retry_cnt = None, 0
        while retry_cnt < self.NUM_RETRIES:
            try:
                audio_features = self.featurizer.process(audio_filepath)
            except Exception:
                print(f"ATTEMPT {retry_cnt} LOAD {audio_filepath} FAILED")
                pass

            if audio_features is not None:
                return audio_features

            retry_cnt += 1

        raise RuntimeError(f"Cannot load {audio_filepath}")

    def __getitem__(
            self, idx: int,
        ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.List[torch.Tensor], tp.List[tp.Union[tp.List, str]]]:
        sample = super().__getitem__(idx)

        tokens_sample = self._read_binary_tokenized_sample(sample, "tokens")
        labels_sample = self._read_binary_tokenized_sample(sample, "labels")

        spectrograms, audio_filepaths = [], []

        if "audios" in sample:
            encoded_audio_filepaths = self._read_binary_tokenized_sample(sample, "audios")
            audio_filepaths = self._get_decoded_str_from_deserialized_bytes(encoded_audio_filepaths)

        elif "files" in sample:
            files = json.loads(sample["files"].decode("utf-8", errors='strict'))
            audio_filepaths = [file["path"] for file in files if file["embed"] and file["type"] == "audio"]

        for audio_filepath in audio_filepaths:
            spectrogram = self._get_spectrogram_tensor_from_filepath(audio_filepath)
            assert spectrogram.shape[0] > 0, f"Incorrect spectrogram: {audio_filepath}"
            spectrograms.append(spectrogram)

        return tokens_sample, labels_sample, spectrograms, audio_filepaths


class PaddingCollatorWrapper:
    SPECTROGRAM_KEYS = ["spectrograms", "spectrogram_lengths"]
    """
    A collator that pads input sequences to a uniform length.

    Args:
    pad_token (int): The pad token ID used for padding sequences.
    max_seq_len (int): The maximum sequence length.

    Attributes:
    pad_token_id (int): Stores the pad token ID.
    max_seq_len (int): Stores the maximum sequence length.
    """
    def __init__(
        self,
        pad_token: int,
        max_seq_len: int,
        batch_mode: tp.Optional[str] = None,
        left_padding: bool = False,
        use_wav_filepaths: bool = False,
    ):
        self.pad_token_id = pad_token
        self.max_seq_len = max_seq_len
        self.batch_mode = batch_mode
        self.left_padding = left_padding
        self.use_wav_filepaths = use_wav_filepaths

    def __call__(self, examples: tp.List[tp.Any]) -> tp.Dict[str, torch.Tensor]:
        batch = {
            "input_ids": [],
            "labels": [],
        }
        if self.use_wav_filepaths:
            batch.update({"wav_filepaths": []})
        batch.update({k: None for k in self.SPECTROGRAM_KEYS})

        sequence_lengths = []
        spectrogram_lengths = []
        spectrogram_size = None
        input_pad_sizes = []

        for input_ids, labels, spectrograms, audio_filepaths in examples:
            assert len(input_ids) == len(labels)
            assert len(input_ids) <= self.max_seq_len
            sequence_lengths.append(len(input_ids))
            # spectrogram : T x F

            assert len(spectrograms) == len(audio_filepaths)

            for spectrogram, audio_filepath in zip(spectrograms, audio_filepaths):
                if spectrogram is not None:
                    assert audio_filepath is not None
                    spectrogram_lengths.append(spectrogram.size(0))
                    spectrogram_size = spectrogram.size(-1)

        assert max(sequence_lengths) <= self.max_seq_len
        local_max_seq_len = max(sequence_lengths)

        # if we have any spectrograms in batch
        if len(spectrogram_lengths) > 0:
            assert spectrogram_size is not None

            B = len(spectrogram_lengths)
            L = max(spectrogram_lengths)
            padded_spectrograms = torch.zeros((B, L, spectrogram_size))

        acoustic_idx = 0
        for e, (input_ids, labels, spectrograms, audio_filepaths) in enumerate(examples):
            pad_size = max(0, local_max_seq_len - len(input_ids))
            input_pad_sizes.append(pad_size)
            pad_size = (pad_size, 0) if self.left_padding else (0, pad_size)
            input_ids = torch.nn.functional.pad(input_ids, pad_size, value=self.pad_token_id)
            labels = torch.nn.functional.pad(labels, pad_size, value=-100)

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            for spectrogram, audio_filepath in zip(spectrograms, audio_filepaths):
                if spectrogram is not None:
                    assert audio_filepath is not None
                    if self.use_wav_filepaths:
                        batch["wav_filepaths"].append(audio_filepath)
                    padded_spectrograms[acoustic_idx, :spectrogram.size(0), :] = spectrogram
                    acoustic_idx += 1

        assert acoustic_idx == len(spectrogram_lengths)

        if len(spectrogram_lengths) > 0:
            batch["spectrograms"] = padded_spectrograms
            batch["spectrogram_lengths"] = torch.tensor(spectrogram_lengths)
            assert padded_spectrograms.shape[0] == len(spectrogram_lengths)
            if self.use_wav_filepaths:
                assert padded_spectrograms.shape[0] == len(batch["wav_filepaths"])

        batch["input_ids"] = torch.vstack(batch["input_ids"])
        batch["labels"] = torch.vstack(batch["labels"])
        batch["pad_sizes"] = torch.tensor(input_pad_sizes)

        if self.batch_mode is not None:
            batch["mode"] = self.batch_mode

        return batch


def build_audio_dataloader(
    cfg: DictConfig,
    device_batch_size: int,
    batch_mode: tp.Optional[str] = None,
    left_padding: bool = False,
    use_wav_filepaths: bool = True,
):
    streams_dict = cfg.dataset.pop("streams", None)
    feature_extractor_cfg = cfg.get("feature_extractor_cfg", None)
    max_seq_len = cfg.dataset.pop("max_seq_len", None)

    # build streams
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            streams.append(Stream(**stream))

    dataset = StreamingAudioDataset(
        streams=streams,
        feature_extractor_cfg=feature_extractor_cfg,
        **cfg.dataset,
    )

    collate_fn = PaddingCollatorWrapper(
        pad_token=0,
        max_seq_len=max_seq_len,
        batch_mode=batch_mode,
        left_padding=left_padding,
        use_wav_filepaths=use_wav_filepaths,
    )

    return DataLoader(dataset,
                      collate_fn=collate_fn,
                      batch_size=device_batch_size,
                      drop_last=cfg.drop_last,
                      num_workers=cfg.num_workers,
                      pin_memory=cfg.get("pin_memory", True),
                      prefetch_factor=cfg.get("prefetch_factor", 2),
                      persistent_workers=cfg.get("persistent_workers", True),
                      timeout=cfg.get("timeout", 0))


def wrap_dataloader(dataloader: torch.utils.data.DataLoader, audio_token_id: int):

    def get_num_samples_in_batch(batch: tp.Mapping) -> int:
        assert isinstance(batch, dict)
        tokens_size = batch["input_ids"].shape[0]
        labels_size = batch["labels"].shape[0]
        assert tokens_size == labels_size
        return tokens_size

    def split_batch(batch: tp.Mapping, microbatch_size: int):
        input_ids_split = batch["input_ids"].split(microbatch_size)
        labels_split = batch["labels"].split(microbatch_size)
        pad_sizes_split = batch["pad_sizes"].split(microbatch_size)

        if batch.get("spectrograms", None) is not None:
            spectrogram_split_ids = []
            for input_ids_chunk in input_ids_split:
                num_audio_tokens = (input_ids_chunk == audio_token_id).sum()
                spectrogram_split_ids.append(num_audio_tokens.item())

            assert sum(spectrogram_split_ids) == batch["spectrograms"].shape[0], \
                f"{spectrogram_split_ids}, {batch['spectrograms'].shape[0]}"

            spectrograms_split = list(batch["spectrograms"].split(spectrogram_split_ids))
            spectrogram_lengths_split = list(batch["spectrogram_lengths"].split(spectrogram_split_ids))

            for i in range(len(spectrograms_split)):
                if spectrograms_split[i].shape[0] == 0:
                    spectrograms_split[i] = None
                    spectrogram_lengths_split[i] = None
                else:
                    # delete exstra padding from spectrogram
                    real_spectrogram_split_length = torch.max(spectrogram_lengths_split[i])
                    spectrograms_split[i] = spectrograms_split[i][:, :real_spectrogram_split_length, :]
                    assert real_spectrogram_split_length == spectrograms_split[i].size(1)
        else:
            spectrograms_split = [None for _ in range(len(input_ids_split))]
            spectrogram_lengths_split = [None for _ in range(len(input_ids_split))]

        mode = batch.get("mode", None)

        return [{
            "input_ids": input_ids_split[i],
            "labels": labels_split[i],
            "pad_sizes": pad_sizes_split[i],
            "spectrograms": spectrograms_split[i],
            "spectrogram_lengths": spectrogram_lengths_split[i],
            "mode": mode,
            "wav_filepaths": (
                batch["wav_filepaths"][i*microbatch_size:(i+1)*microbatch_size]
                if batch.get("wav_filepaths") and i < len(batch["wav_filepaths"])
                else None
            )
        } for i in range(len(input_ids_split))]

    return DataSpec(
        dataloader=dataloader,
        split_batch=split_batch,
        get_num_samples_in_batch=get_num_samples_in_batch
    )


def build_evaluator(cfg: DictConfig, dataset_name: str, tokenizer, model, audio_token_id: int):
    eval_loader_cfg = cfg["eval_loaders"][dataset_name]
    eval_loader_cfg = om.merge(cfg.default_eval_loader_params, eval_loader_cfg)
    eval_loader_cfg.dataset = om.merge(cfg.default_eval_dataset_params, eval_loader_cfg.dataset)

    metric_names = list(eval_loader_cfg.pop("metric_names", model.val_metrics.keys()))
    batch_mode = eval_loader_cfg.pop("mode", None)
    left_padding = eval_loader_cfg.pop("left_padding", False)
    subset_num_batches = eval_loader_cfg.pop("subset_num_batches", None)
    use_wav_filepaths = eval_loader_cfg.pop("use_wav_filepaths", False)

    eval_loader = build_audio_dataloader(
        eval_loader_cfg,
        cfg.device_eval_batch_size,
        batch_mode,
        left_padding,
        use_wav_filepaths,
    )

    eval_loader = wrap_dataloader(eval_loader, audio_token_id)
    eval_loader = Evaluator(
        label=dataset_name,
        dataloader=eval_loader,
        metric_names=metric_names,
        subset_num_batches=subset_num_batches
    )

    return eval_loader
