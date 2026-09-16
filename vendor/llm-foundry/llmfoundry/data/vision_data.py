import io
import json
import torch
import tarfile
import warnings
import logging

from typing import List, Any, Dict
from pathlib import Path
from PIL import Image

from streaming import Stream
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from composer.utils import dist

from llmfoundry.data.text_data import FlashAttnCollatorWrapper, StreamingTextDataset
from llmfoundry.data.processors import GigaVisionProcessor
from llmfoundry.data.processors.mm_utils import get_crop_size
from llmfoundry.data.data import read_binary_tokenized_sample

logger = logging.getLogger(__name__)


class StreamingVisionDataset(StreamingTextDataset):
    def __init__(
        self,
        processor_config: Dict[str, Any],
        files_folder_path: List[str] = [],
        dpo_max_expanded_len: int = 0,
        dpo_resample_attempts: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.processor = GigaVisionProcessor(**processor_config)
        self.files_folder_path = [
            Path(path) if isinstance(path, str) else path for path in files_folder_path
        ]
        self.dpo_max_expanded_len = dpo_max_expanded_len
        self.dpo_resample_attempts = dpo_resample_attempts

    def __getitem__(self, idx: int):
        dpo_mode = False
        sample = super(StreamingTextDataset, self).__getitem__(idx)

        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "tokens" in sample:
            token_sample = read_binary_tokenized_sample(self.max_seq_len, sample)
        elif "ab_ids" in sample:
            token_sample = self._read_dpo_binary_tokenized_sample(sample)
            dpo_mode = True
        else:
            raise RuntimeError(
                "StreamingTextDataset needs samples to have a `text`, `tokens`, or `ab_ids` column"
            )

        if self.sft_mode:
            labels_sample = read_binary_tokenized_sample(
                self.max_seq_len, sample, "labels"
            )

        stream_idx = self._get_stream_index(idx)

        if self.files_folder_path[stream_idx] is None:
            assert self.sft_mode, "S3 streaming requires sft_mode=True"
            payload = sample.get("payloads")
            assert isinstance(payload, bytes), (
                f"Expected payloads to be bytes, got {type(payload)}"
            )

            files = json.loads(sample["files"].decode("utf-8", errors="strict"))
            name_to_image = self._get_pil_images(payloads=payload)

            pil_images = []
            fallback_count = 0
            for f in files:
                image = name_to_image.get(f["name"])
                if image is None:
                    logger.warning(
                        f"Missing image {f['name']} in tar payload, fallback to black image"
                    )
                    image = self._get_black_image()
                    fallback_count += 1
                pil_images.append(image)

            processed_data = self.processor.preprocess(
                pil_images=pil_images,
                input_ids=token_sample,
                labels=labels_sample,
                return_dict=True,
            )
            processed_data["processor_fallback_count"] += fallback_count
            return processed_data

        image_paths = self._get_image_paths(sample, stream_idx)

        if self.sft_mode:
            return self.processor.preprocess(
                image_paths, token_sample, labels_sample, return_dict=True
            )
        elif dpo_mode:
            token_sample["q_ids"], _, token_sample["q_mask"], tensor_images = (
                self.processor.preprocess(
                    image_paths=image_paths,
                    input_ids=token_sample["q_ids"],
                    attention_mask=token_sample["q_mask"],
                    mask_fill_value=0,
                )
            )
            token_sample["tensor_images"] = tensor_images

            # Skip samples that are too long after image token expansion
            if self.dpo_max_expanded_len > 0:
                expanded_len = max(
                    len(token_sample["q_ids"]) + len(token_sample["ab_ids"]),
                    len(token_sample["q_ids"]) + len(token_sample["aw_ids"]),
                )
                if expanded_len > self.dpo_max_expanded_len:
                    return self._dpo_resample(idx, expanded_len)

            return token_sample
        else:
            raise RuntimeError(
                "Use sft_mode=True or provide DPO data with 'ab_ids' in sample"
            )

    _dpo_resample_depth = 0

    def _dpo_resample(self, failed_idx: int, failed_len: int):
        """Skip to next index. Recursion via __getitem__ handles consecutive long samples."""
        self._dpo_resample_depth += 1
        try:
            if self._dpo_resample_depth > self.dpo_resample_attempts:
                raise RuntimeError(
                    f"DPO resample: exhausted {self.dpo_resample_attempts} attempts "
                    f"starting from idx={failed_idx} (len={failed_len} > {self.dpo_max_expanded_len}). "
                    f"Too many consecutive long samples — increase dpo_resample_attempts or "
                    f"filter long samples at tokenization time."
                )
            warnings.warn(
                f"DPO resample: idx={failed_idx} too long "
                f"({failed_len} > {self.dpo_max_expanded_len}), "
                f"trying idx={failed_idx + 1} (depth {self._dpo_resample_depth})"
            )
            next_idx = (failed_idx + 1) % len(self)
            return self.__getitem__(next_idx)
        finally:
            self._dpo_resample_depth -= 1

    def _get_image_paths(self, sample, stream_idx):
        image_paths = []
        if "images" in sample:
            if isinstance(sample["images"], str):
                image_path = sample["images"]
            else:
                deserialized_image_path = read_binary_tokenized_sample(
                    self.max_seq_len, sample, "images"
                )
                image_path = self._get_decoded_str_from_deserialized_bytes(
                    deserialized_image_path
                )

            image_paths = (
                list(image_path.strip().split("\n")) if image_path != "" else []
            )
        elif "files" in sample:
            files = json.loads(sample["files"].decode("utf-8", errors="strict"))
            image_paths = [
                file["path"]
                for file in files
                if file["embed"] and file["type"] == "image"
            ]
        elif "ab_files" in sample:
            # DPO mode: images are in context (shared by chosen/rejected)
            files = json.loads(sample["ab_files"].decode("utf-8", errors="strict"))
            image_paths = [
                file["path"]
                for file in files
                if file.get("embed") and file.get("type") == "image"
            ]

        image_paths = [
            Path(self.files_folder_path[stream_idx]) / str(path).lstrip("/")
            for path in image_paths
        ]
        return image_paths

    @staticmethod
    def _get_pil_images(payloads: bytes) -> Dict[str, Image.Image]:
        name_to_image = dict()
        with tarfile.open(fileobj=io.BytesIO(payloads), mode="r:*") as tar:
            for member in tar.getmembers():
                extracted = tar.extractfile(member)
                image = Image.open(io.BytesIO(extracted.read()))
                image.load()
                name_to_image[member.name] = image
        return name_to_image

    def _get_black_image(self) -> Image.Image:
        size = get_crop_size(self.processor.image_processor)
        return Image.new("RGB", (size, size))

    def _get_stream_index(self, idx: int) -> int:
        shard_idx = self.spanner[idx][0]
        stream_idx = self.stream_per_shard[shard_idx]
        return stream_idx

    def _get_decoded_str_from_deserialized_bytes(self, deserialized_bytes):
        decoded_string = "".join([chr(code) for code in deserialized_bytes]).strip()
        return decoded_string


class PaddingCollatorWrapper:
    """
    A collator that pads input sequences to a uniform length.

    Args:
    pad_token (int): The pad token ID used for padding sequences.
    max_seq_len (int): The maximum sequence length.

    Attributes:
    pad_token_id (int): Stores the pad token ID.
    max_seq_len (int): Stores the maximum sequence length.
    """

    def __init__(self, pad_token: int, max_seq_len: int, scale: float = 1.0):
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
        """
        Collates batch of examples by padding them to the maximum sequence length in the batch.

        Args:
        examples (List[Any]): A list of dictionaries, each containing input_ids, labels, images and processor_fallback_count for
            a single example.

        Returns:
        Dict[str, torch.Tensor]: A dictionary containing padded tensors for "input_ids", "labels",
            and "images".
        """
        batch = {
            "input_ids": [],
            "labels": [],
            "images": [],
        }
        local_max_seq_len = 0
        max_num_images = 0
        total_processor_fallback_count = 0

        for example in examples:
            input_ids = example["input_ids"]
            labels = example["labels"]
            images = example["images"]

            assert len(input_ids) == len(labels)
            assert len(input_ids) <= self.max_seq_len

            local_max_seq_len = max(local_max_seq_len, len(input_ids))
            max_num_images = max(max_num_images, images.size(0))
            total_processor_fallback_count += example["processor_fallback_count"]

        if local_max_seq_len % self.divider != 0:
            local_max_seq_len = self.divider * (local_max_seq_len // self.divider + 1)

        assert local_max_seq_len <= self.max_seq_len

        for example in examples:
            input_ids = example["input_ids"]
            labels = example["labels"]
            images = example["images"]

            pad_size = max(0, local_max_seq_len - len(input_ids))
            batch["input_ids"].append(
                torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=self.pad_token_id
                )
            )
            batch["labels"].append(
                torch.nn.functional.pad(labels, (0, pad_size), value=-100)
            )

            # composer.trainer требует чтобы первая размерность в тензоре была размер батча.
            # поэтому мы падим их нулевым тензором плюс упаковываем через stack
            if images.size(0) < max_num_images:
                padding_images = torch.zeros(
                    (
                        max_num_images - images.size(0),
                        images.size(1),
                        images.size(2),
                        images.size(3),
                    ),
                    dtype=images.dtype,
                    device=images.device,
                )
                images = torch.cat((images, padding_images), dim=0)
            batch["images"].append(images)

        for k in batch.keys():
            if k == "images":
                # [num_samples_in_batch, num_images, num_channels, width, height]
                batch[k] = torch.stack(batch[k])
            else:
                batch[k] = torch.vstack(batch[k])

        batch["local_max_seq_len"] = local_max_seq_len
        batch["vision_processor_fallback_count"] = total_processor_fallback_count

        return batch


class DpoMultimodalCollatorWrapper:
    """Collator for DPO training with multimodal (vision) data.

    Produces batches with chosen/rejected pairs and images in the format
    expected by ComposerDPOModel:
        input_ids:      [batch_size, 2, seq_len]
        attention_mask: [batch_size, 2, seq_len]
        action_mask:    [batch_size, 2, seq_len]
        margins:        [batch_size]
        images:         [batch_size, 2*K, C, H, W]  (chosen + rejected crops concatenated)
    """

    def __init__(
        self, max_token_len, pad_token_id, image_token_id, pad_to_max_len=False
    ):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id if pad_token_id is not None else 2
        self.pad_to_max_len = pad_to_max_len
        self.image_token_id = image_token_id
        tp_sp_group_size = dist.get_tp_sp_group_size()
        self.divider = tp_sp_group_size if tp_sp_group_size is not None else 8
        self.divider *= 2
        print(
            f"[INFO]: Doubled divider in DpoCollatorWrapper (from {self.divider // 2} to {self.divider})",
            "to make up for splitting the seq in two halves in the end",
        )

    def __call__(self, data: List[Any]) -> Dict[str, torch.Tensor]:
        inputs = [torch.cat((s["q_ids"], s["ab_ids"])) for s in data] + [
            torch.cat((s["q_ids"], s["aw_ids"])) for s in data
        ]

        for s in inputs:
            assert len(s) <= self.max_token_len, (
                f"DPO sequence length {len(s)} exceeds max_token_len {self.max_token_len}. "
            )

        inputs = pad_sequence(inputs, padding_value=self.pad_token_id, batch_first=True)

        act_mask = [torch.cat((s["q_mask"], s["ab_mask"])) for s in data] + [
            torch.cat((s["q_mask"], s["aw_mask"])) for s in data
        ]
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

        images = [s["tensor_images"] for s in data]
        max_num_images = max(image.size(0) for image in images)
        batch_images = []
        for image in images:
            if image.size(0) < max_num_images:
                padding = torch.zeros(
                    (
                        max_num_images - image.size(0),
                        image.size(1),
                        image.size(2),
                        image.size(3),
                    ),
                    dtype=image.dtype,
                    device=image.device,
                )
                image = torch.cat((image, padding), dim=0)
            batch_images.append(image)

        images = torch.stack(batch_images)
        # Duplicate images for chosen and rejected (same question, same images).
        # Result: [B, 2*K, C, H, W] where first K are chosen, last K are rejected.
        images = torch.cat((images, images), dim=1)

        return {
            "input_ids": inputs.reshape(
                2, inputs.shape[0] // 2, inputs.shape[1]
            ).transpose(0, 1),
            "attention_mask": inputs.not_equal(self.pad_token_id)
            .long()
            .reshape(2, inputs.shape[0] // 2, inputs.shape[1])
            .transpose(0, 1),
            "action_mask": act_mask.reshape(
                2, act_mask.shape[0] // 2, act_mask.shape[1]
            ).transpose(0, 1),
            "margins": margins,
            "images": images,
        }


def build_vision_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
):
    assert cfg.name == "vision", (
        f"Tried to build image dataloader with cfg.name={cfg.name}"
    )

    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset_hf.py"
        )

    streams_dict = cfg.dataset.pop("streams", None)
    eos_token_id = cfg.get("eos_token_id", None)

    sft_mode = cfg.get("sft_mode", False)
    padding_scale = cfg.get("padding_scale", 1.0)

    streams = None
    files_folder_path = []
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            assert "files_folder_path" in stream or "remote" in stream, (
                "check field 'files_folder_path' or 'remote' (to use s3) in stream config. 'image_folder_path' is deprecated"
            )
            if "files_folder_path" in stream:
                files_folder_path.append(stream.pop("files_folder_path"))
            else:
                files_folder_path.append(None)
            streams.append(Stream(**stream))

    dataset = StreamingVisionDataset(
        tokenizer=tokenizer,
        streams=streams,
        cache_limit="256gb",
        sft_mode=sft_mode,
        batch_size=device_batch_size,
        files_folder_path=files_folder_path,
        dpo_max_expanded_len=cfg.dataset.max_seq_len if cfg.get("dpo_data") else 0,
        dpo_resample_attempts=cfg.get("dpo_resample_attempts", 50),
        **cfg.dataset,
    )

    collate_fn = PaddingCollatorWrapper(
        pad_token=0,
        max_seq_len=cfg.dataset.max_seq_len,
        scale=padding_scale,
    )

    if cfg.get("batch_type", None) == "flash_attn" and not cfg.get("dpo_data", None):
        collate_fn = FlashAttnCollatorWrapper(
            base_collator=collate_fn,
            eos_token=eos_token_id,
        )

    if cfg.get("dpo_data", None):
        collate_fn = DpoMultimodalCollatorWrapper(
            max_token_len=cfg.dataset.max_seq_len,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else eos_token_id,
            pad_to_max_len=cfg.get("pad_to_max_len", False),
            image_token_id=cfg.dataset.processor_config.image_token_id,
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
