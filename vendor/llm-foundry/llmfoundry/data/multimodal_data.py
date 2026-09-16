from typing import List, Dict, Callable
import inspect
from omegaconf import DictConfig

import json
from streaming import Stream
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.text_data import StreamingTextDataset
from llmfoundry.data import processors, collators
from llmfoundry.data.data import read_binary_tokenized_sample


class StreamingMMDataset(StreamingTextDataset):
    MODALITY_TENSOR_KEYS = {
        "image": "images",
        "video": "tensor_frames",
    }

    def __init__(self, processors=None, **kwargs):
        super().__init__(**kwargs)
        self.processors = processors

    def __getitem__(self, idx: int):
        sample = super(StreamingTextDataset, self).__getitem__(idx)

        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "tokens" in sample:
            token_sample = read_binary_tokenized_sample(self.max_seq_len, sample)
        else:
            raise RuntimeError(
                "StreamingTextDataset needs samples to have a `text` or `tokens` column"
            )

        labels_sample = read_binary_tokenized_sample(self.max_seq_len, sample, "labels")

        files = []
        if "files" in sample:
            files = json.loads(sample["files"].decode("utf-8", errors="strict"))

        sample_dict = {
            "text": {
                "input_ids": token_sample,
                "labels": labels_sample,
            }
        }

        stream = self._get_stream(idx)
        for processor in self.processors:
            file_paths = [
                Path(stream.files_folder_path) / str(file["path"]).lstrip("/")
                for file in files
                if file["embed"] and file["type"] == processor.modality
            ]
            preprocessed_sample = processor.preprocess(
                file_paths,
                sample_dict["text"]["input_ids"],
                sample_dict["text"]["labels"],
                return_dict=True,
            )

            sample_dict["text"].update(
                {
                    "input_ids": preprocessed_sample["input_ids"],
                    "labels": preprocessed_sample["labels"],
                }
            )
            tensor_key = self.MODALITY_TENSOR_KEYS[processor.modality]
            sample_dict[processor.modality] = {
                "pixel_values": preprocessed_sample[tensor_key],
                "processor_fallback_count": preprocessed_sample[
                    "processor_fallback_count"
                ],
            }

        return sample_dict

    def _get_stream(self, idx):
        stream_idx = self._get_stream_idx(idx)
        return self.streams[stream_idx]

    def _get_stream_idx(self, idx):
        shard_idx = self.spanner[idx][0]
        stream_idx = self.stream_per_shard[shard_idx]
        return stream_idx


class MMStream(Stream):
    def __init__(
        self,
        name: str = None,
        modality: str = "vision",
        files_folder_path: str = "path",
        **kwargs,
    ):
        self.name = name
        self.modality = modality
        self.files_folder_path = files_folder_path
        self.stream_kwargs = self._kwargs_filter(kwargs)

        super().__init__(**self.stream_kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _kwargs_filter(self, kwargs):
        """
        Extracts from the passed `kwargs` dictionary only those keys that are valid
        arguments for the `Stream` constructor.

        Args:
            kwargs (Dict): Dictionary of keyword arguments.

        Returns:
            Dict: Dictionary containing only arguments valid for the `Stream` constructor.
        """

        stream_kwargs = list(inspect.signature(Stream.__init__).parameters.keys())[1:]
        stream_kwargs = {key: kwargs.pop(key, None) for key in stream_kwargs}
        return stream_kwargs

    def __repr__(self):
        repr = ""
        repr += f"Stream name: {self.name}\n"
        repr += f"Modality: {self.modality}\n"
        repr += f"Files folder path: {self.files_folder_path}\n"
        repr += "\n".join(
            [f"{key}: {value}" for key, value in self.stream_kwargs.items()]
        )
        return repr


class MMCollator:
    """
    A collator that pads input sequences to a uniform length.

    Args:
    pad_token (int): The pad token ID used for padding sequences.
    max_seq_len (int): The maximum sequence length.

    Attributes:
    pad_token_id (int): Stores the pad token ID.
    max_seq_len (int): Stores the maximum sequence length.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.modality_collators = {}

    def register_collator(self, modality: str, collator_fn: Callable):
        self.modality_collators[modality] = collator_fn

    def __call__(self, examples: List[Dict]):
        batch = {}

        # Process only registered modalities
        for modality in self.modality_collators:
            data_list = [ex[modality] for ex in examples if modality in ex]

            if data_list:
                assert len(data_list) == self.batch_size, (
                    f"Modality {modality} present in {len(data_list)} examples, "
                    f"expected {self.batch_size}"
                )

                collated_data = self.modality_collators[modality](data_list)
                batch.update(collated_data)

        batch["local_max_seq_len"] = batch["input_ids"].shape[-1]
        return batch


def build_processor(processor_config, tokenizer=None):
    cls_name = processor_config.pop("name")

    Processor = getattr(processors, cls_name)
    print(processor_config)

    if cls_name == "GigaVideoProcessor":
        return Processor(**processor_config, tokenizer=tokenizer)
    else:
        return Processor(**processor_config)


def build_collators(collator_config):
    Collator = getattr(collators, collator_config.pop("name"))

    return Collator(**collator_config)


def build_multimodal_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
):
    assert cfg.name == "multimodal", (
        f"Tried to build multimodal dataloader with cfg.name={cfg.name}"
    )

    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset_hf.py"
        )

    streams_dict = cfg.dataset.pop("streams", None)
    padding_scale = cfg.get("padding_scale", 1.0)
    sft_mode = cfg.get("sft_mode", False)
    assert sft_mode, "Only sft_mode=True is supported for multimodal dataloader"

    streams = None
    if streams_dict is not None:
        streams = [
            MMStream(name=stream, **streams_dict[stream]) for stream in streams_dict
        ]

    processors_config = cfg.dataset.pop("processors")
    processors = [
        build_processor(processor_config, tokenizer=tokenizer)
        for processor_config in processors_config
    ]

    collators_config = cfg.dataset.pop("collators")
    collate_fn = MMCollator(batch_size=device_batch_size)
    collate_fn.register_collator(
        "text",
        collators.TextCollator(
            pad_token_id=0, max_seq_len=cfg.dataset.max_seq_len, scale=padding_scale
        ),
    )
    for collator_config in collators_config:
        collate_fn.register_collator(
            collator_config.pop("modality"), build_collators(collator_config)
        )

    dataset = StreamingMMDataset(
        processors=processors,
        tokenizer=tokenizer,
        streams=streams,
        cache_limit="256gb",
        sft_mode=sft_mode,
        **cfg.dataset,
    )

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )
