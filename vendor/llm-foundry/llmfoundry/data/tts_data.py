from collections import defaultdict
import itertools
from typing import List, Any, Dict, Tuple, Optional, Union
import traceback
import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path
from streaming import Stream
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging

from composer.utils import dist

from llmfoundry.data.text_data import StreamingTextDataset
from .text_data import FlashAttnCollatorWrapper


from typing import Optional, List
import inspect
from dataclasses import dataclass, asdict

logger = logging.get_logger(__name__)


class ModalityDataConfig:
    ignore_token_id: int = -100

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(**{
            k: v for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        })

    def dict(self):
        return asdict(self)


@dataclass
class TextTokensDataConfig(ModalityDataConfig):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


@dataclass
class SpeechTokensDataConfig(ModalityDataConfig):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    q_ids: List[List[int]]

    dtm: bool = True
    n_q: int = 1
    audio_delay: int = 0
    n_frames_prediction: int = 1
    dropout: float = 0.
    mask_token_id: Optional[int] = None

    def __post_init__(self):
        if not self.dtm:
           self.q_ids = [list(itertools.chain.from_iterable(self.q_ids))]
        if self.q_ids and sum([len(_) for _ in self.q_ids]) != self.n_q:
            self.n_q = sum([len(_) for _ in self.q_ids])


class StreamingTTSDataset(StreamingTextDataset):
    def __init__(
        self,
        max_seq_len: int,
        code_dataset: List[str],

        speech_modality: Dict,
        text_modality: Dict,

        tasks: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(max_seq_len=max_seq_len, **kwargs)
        self.max_seq_len = max_seq_len
        self.code_dataset = code_dataset
        self.tasks = tasks or ["tts"] * len(self.streams)

        self.sp_cfg = SpeechTokensDataConfig.from_kwargs(**speech_modality)
        self.te_cfg = TextTokensDataConfig.from_kwargs(**text_modality)

    def _get_stream_index(self, idx: int) -> int:
        shard_idx = self.spanner[idx][0]
        stream_idx = int(self.stream_per_shard[shard_idx])
        return stream_idx

    def _get_decoded_str_from_deserialized_bytes(self, deserialized_bytes: torch.Tensor):
        decoded_strings = "".join([chr(code) for code in deserialized_bytes.tolist()]).strip().split("\n")
        return decoded_strings

    @staticmethod
    def _read_np_array(path: Union[str, Path]) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).squeeze()

    @staticmethod
    def _insert(place: torch.Tensor, value: torch.Tensor, idx: int) -> torch.Tensor:
        place = torch.cat([place[..., :idx], value, place[..., idx:]], dim=(place.ndim - 1))
        return place

    def _code_root(self, sample_id: int) -> Path:
        stream_id = self._get_stream_index(sample_id)
        code_root = Path(self.code_dataset[stream_id])
        return code_root

    @staticmethod
    def get_code(code_root: Path, path: Path) -> torch.Tensor:
        if path.as_posix().startswith(code_root.parent.as_posix()):
            path = Path(*path.parts[len(code_root.parent.parts):])
        path = path.with_suffix(".npy")
        code = StreamingTTSDataset._read_np_array(code_root / path)
        return code

    @staticmethod
    def _schedule_rvq_tokens(
            tensor: torch.Tensor, pad_token_id: int, ignore_token_id: int,
            q_ids: List[List[int]]
        ) -> torch.Tensor:
        speech_tokens_per_frame = sum([len(_) for _ in q_ids])
        speech_len = tensor.shape[1] + len(q_ids) - 1

        tokens = torch.full((speech_tokens_per_frame, speech_len), pad_token_id, dtype=tensor.dtype)

        t_id, shift = 0, 0
        for q_sub in q_ids:
            for idx, q_idx in enumerate(q_sub):
                tokens[t_id + idx, shift : shift + tensor.shape[1]] = tensor[q_idx]
            t_id += len(q_sub)
            shift += 1
        return tokens

    @staticmethod
    def _prepare_code(code: torch.Tensor, sp_cfg: SpeechTokensDataConfig) -> torch.Tensor:
        code = F.pad(code, (1, 0), value=sp_cfg.bos_token_id)
        code = F.pad(code, (0, 1), value=sp_cfg.eos_token_id)

        code_len = math.ceil(code.shape[1] // sp_cfg.n_frames_prediction) * sp_cfg.n_frames_prediction
        code = F.pad(code, (0, code_len - code.shape[1]), value=sp_cfg.pad_token_id)
        code = rearrange(code, "q (t nf) -> (q nf) t", nf=sp_cfg.n_frames_prediction)

        if sp_cfg.dtm:
            code_tokens = StreamingTTSDataset._schedule_rvq_tokens(
                code,
                sp_cfg.pad_token_id,
                sp_cfg.ignore_token_id,
                q_ids=sp_cfg.q_ids
            )
        else:
            code_tokens = code[torch.tensor(sp_cfg.q_ids).squeeze()]

        return code_tokens

    @staticmethod
    def _find_and_delete_special_tokens(
            tokens: torch.Tensor, labels: torch.Tensor, bos_token_id: int, eos_token_id: int
        ) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
        bos_positions = sorted(torch.nonzero(tokens == bos_token_id).flatten().tolist())
        eos_positions = sorted(torch.nonzero(tokens == eos_token_id).flatten().tolist())
        assert len(bos_positions) == len(eos_positions) or \
            len(bos_positions) == len(eos_positions) + 1, (len(bos_positions), len(eos_positions))

        labels = labels[(tokens != bos_token_id) & (tokens != eos_token_id)]
        tokens = tokens[(tokens != bos_token_id) & (tokens != eos_token_id)]

        positions = list(itertools.chain.from_iterable(zip(bos_positions, eos_positions)))
        positions = np.array(positions)
        assert np.all(positions[1:] >= positions[:-1])
        positions = positions - np.arange(positions.shape[0])

        bos_positions = [positions[i] for i in range(0, len(positions), 2)]
        eos_positions = [positions[i] for i in range(1, len(positions), 2)]

        return bos_positions, eos_positions, tokens, labels

    @staticmethod
    def _get_text_sample(
            tokens_sample: torch.Tensor, labels_sample: torch.Tensor, sp_cfg: SpeechTokensDataConfig
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_input_ids = tokens_sample
        t_labels = labels_sample
        s_input_ids = torch.full(
            (sp_cfg.n_q * sp_cfg.n_frames_prediction, t_input_ids.shape[0]),
            sp_cfg.pad_token_id
        )
        s_labels = torch.full(
            (sp_cfg.n_q * sp_cfg.n_frames_prediction, t_input_ids.shape[0]),
            sp_cfg.ignore_token_id
        )
        return t_input_ids, t_labels, s_input_ids, s_labels

    @staticmethod
    def _combine_text_and_codes(
            t_input_ids: torch.Tensor,
            t_labels: torch.Tensor,
            codes: List[torch.Tensor],
            bos_texts: List[int],
            eos_texts: List[int],
            te_cfg: TextTokensDataConfig,
            sp_cfg: SpeechTokensDataConfig,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # bos_texts - positions of first text answer tokens
        # eos_texts - positions of <|mesage_sep|> token anfter text answer

        n_speech_t_per_step = sp_cfg.n_q * sp_cfg.n_frames_prediction
        s_input_ids = torch.full((n_speech_t_per_step, t_input_ids.shape[0]), sp_cfg.pad_token_id)
        s_labels = torch.full((n_speech_t_per_step, t_input_ids.shape[0]), sp_cfg.ignore_token_id)

        cum_pad_len = 0
        for bos, eos, code in zip(bos_texts, eos_texts, codes):
            len_text = eos - bos
            bos += cum_pad_len
            eos += cum_pad_len

            code_tokens = StreamingTTSDataset._prepare_code(code, sp_cfg)
            code_tokens = F.pad(code_tokens, (sp_cfg.audio_delay, 0, 0, 0), value=sp_cfg.pad_token_id)
            len_diff = code_tokens.shape[-1] - len_text - 1
            if len_diff < 0:
                code_tokens = F.pad(code_tokens, (0, -len_diff, 0, 0), value=sp_cfg.pad_token_id)
            else:
                cum_pad_len += len_diff

                pad = torch.full((len_diff,), te_cfg.pad_token_id)
                t_input_ids = StreamingTTSDataset._insert(t_input_ids, pad, eos + 1)
                pad = torch.full((len_diff,), te_cfg.ignore_token_id)
                t_labels = StreamingTTSDataset._insert(t_labels, pad, eos + 1)
                pad = torch.full((n_speech_t_per_step, len_diff), sp_cfg.pad_token_id)
                s_input_ids = StreamingTTSDataset._insert(s_input_ids, pad, eos + 1)
                s_labels = StreamingTTSDataset._insert(s_labels, pad, eos + 1)

            s_input_ids[:, bos : bos + code_tokens.shape[-1]] = code_tokens
            trainable_code = (t_labels[bos: eos] != te_cfg.ignore_token_id).any()
            if True: #trainable_code:
                s_labels[:, bos : bos + code_tokens.shape[-1]] = code_tokens

            assert t_input_ids.shape[0] == t_labels.shape[0] == s_input_ids.shape[1], \
                (t_input_ids.shape[0], t_labels.shape[0], s_input_ids.shape[1])

        # remove paddings from labels
        s_labels[s_labels == sp_cfg.pad_token_id] = sp_cfg.ignore_token_id

        return t_input_ids, t_labels, s_input_ids, s_labels

    @staticmethod
    def get_tts_sample(tokens_sample: torch.Tensor, labels_sample: torch.Tensor,
            codes: List[torch.Tensor], te_cfg: TextTokensDataConfig, sp_cfg: SpeechTokensDataConfig):
        (bos_texts,
         eos_texts,
         t_input_ids,
         t_labels) = StreamingTTSDataset._find_and_delete_special_tokens(
            tokens_sample,
            labels_sample,
            te_cfg.bos_token_id,
            te_cfg.eos_token_id
        )
        assert  (te_cfg.bos_token_id not in t_input_ids) and (te_cfg.bos_token_id not in t_input_ids)
        assert len(bos_texts) == 1 and len(eos_texts) <= 1, (len(bos_texts), len(eos_texts))

        codes = [torch.cat(codes, 1)]

        return StreamingTTSDataset._combine_text_and_codes(
            t_input_ids, t_labels, codes, bos_texts, eos_texts, te_cfg, sp_cfg
        )

    @staticmethod
    def get_fs_tts_sample(tokens_sample: torch.Tensor, labels_sample: torch.Tensor, codes: List[torch.Tensor], te_cfg: TextTokensDataConfig, sp_cfg: SpeechTokensDataConfig):
        bos_texts, eos_texts, t_input_ids, t_labels = StreamingTTSDataset._find_and_delete_special_tokens(
            tokens_sample,
            labels_sample,
            te_cfg.bos_token_id,
            te_cfg.eos_token_id
        )
        assert  (te_cfg.bos_token_id not in t_input_ids) and (te_cfg.bos_token_id not in t_input_ids)
        assert len(eos_texts) == len(codes), f'len(eos_texts)={len(eos_texts)} & len(codes)={len(codes)}'

        return StreamingTTSDataset._combine_text_and_codes(
            t_input_ids, t_labels, codes, bos_texts, eos_texts, te_cfg, sp_cfg
        )

    def __getitem__(self, sample_id: int):
        try:
            sample = super(StreamingTextDataset, self).__getitem__(sample_id)
            tokens_sample = self._read_binary_tokenized_sample(sample, "tokens")
            labels_sample = self._read_binary_tokenized_sample(sample, "labels")

            task = self.tasks[self._get_stream_index(sample_id)]
            if task == 'text':
                t_input_ids, t_labels, s_input_ids, s_labels = self._get_text_sample(tokens_sample, labels_sample, self.sp_cfg)
            else:
                encoded_paths = self._read_binary_tokenized_sample(sample, "audio")
                code_paths = self._get_decoded_str_from_deserialized_bytes(encoded_paths)
                code_root = self._code_root(sample_id)
                codes = [self.get_code(code_root, Path(path)) for path in code_paths]

                if task == 'tts':
                    t_input_ids, t_labels, s_input_ids, s_labels = self.get_tts_sample(tokens_sample, labels_sample, codes, self.te_cfg, self.sp_cfg)
                elif task == 'qa_t2ts' or task == "fs_tts":
                    t_input_ids, t_labels, s_input_ids, s_labels = self.get_fs_tts_sample(tokens_sample, labels_sample, codes, self.te_cfg, self.sp_cfg)
                else:
                    raise NotImplementedError()

                if (s_labels != self.sp_cfg.ignore_token_id).sum() == 0:
                    raise ValueError("no trainable speech tokens")

            assert len(t_labels) > 0 and len(s_labels) > 0, "no tokens in example"

            if len(t_input_ids) >= self.max_seq_len:
                t_input_ids = t_input_ids[:self.max_seq_len]
                s_input_ids = s_input_ids[:, :self.max_seq_len]
                t_labels = t_labels[:self.max_seq_len]
                s_labels = s_labels[:, :self.max_seq_len]

            if (t_labels != self.te_cfg.ignore_token_id).sum() == 0:
                raise ValueError("no trainable text tokens")

            return t_input_ids, t_labels, s_input_ids, s_labels, task
        except BaseException:
            logger.info(traceback.format_exc())
            return self[(sample_id + 10) % len(self)]


class PaddingCollatorWrapper:
    """
    A collator that pads input sequences to a uniform length.
    """
    def __init__(
        self,
        pad_text_token_id: int,
        pad_speech_token_id: int,
        max_seq_len: int,
        ignore_loss_label_token_id: int = -100,
        collate_attention_mask: bool = True,
    ):
        self.pad_text_token_id = pad_text_token_id
        self.pad_speech_token_id = pad_speech_token_id
        self.max_seq_len = max_seq_len
        self.ignore_loss_label_token_id = ignore_loss_label_token_id
        self.collate_attention_mask = collate_attention_mask
        tp_sp_group_size = dist.get_tp_sp_group_size()
        self.divider = tp_sp_group_size if tp_sp_group_size is not None else 1

    def __call__(self, examples: List[Any], padding_side: str = "right") -> Dict[str, torch.Tensor]:
        """
        Collates batch of examples by padding them to the maximum sequence length in the batch.

        Args:
        examples (List[Any]): A list of tuples, each containing input_ids, labels, and images for
            a single example.

        Returns:
        Dict[str, torch.Tensor]: A dictionary containing padded tensors for "input_ids", "labels",
            and "images".
        """
        batch: Dict = defaultdict(list)
        sizes = []
        for t_input_ids, t_labels, s_input_ids, s_labels, task in examples:
            assert t_input_ids.shape[0] == t_labels.shape[0], f't_input_ids={t_input_ids.shape}, t_labels={t_labels.shape}'
            assert t_input_ids.shape[0] == s_input_ids.shape[-1], f't_input_ids={t_input_ids.shape}, s_input_ids={s_input_ids.shape}'
            assert t_input_ids.shape[0] == s_labels.shape[-1], f't_input_ids={t_input_ids.shape}, s_labels={s_labels.shape}'
            batch["input_ids"].append(t_input_ids)
            batch["labels"].append(t_labels)
            batch["speech_input_ids"].append(s_input_ids.transpose(1, 0))
            batch["speech_labels"].append(s_labels.transpose(1, 0))
            batch["task"].append(task)
            sizes.append(t_input_ids.shape[0])

        batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_side=padding_side, padding_value=self.pad_text_token_id)
        batch["labels"] = pad_sequence(batch["labels"], batch_first=True, padding_side=padding_side, padding_value=self.ignore_loss_label_token_id)
        batch["speech_input_ids"] = pad_sequence(batch["speech_input_ids"], batch_first=True, padding_side=padding_side, padding_value=self.pad_speech_token_id).transpose(1, 2)
        batch["speech_labels"] = pad_sequence(batch["speech_labels"], batch_first=True, padding_side=padding_side, padding_value=self.ignore_loss_label_token_id).transpose(1, 2)

        if self.collate_attention_mask:
            batch["attention_mask"] = (batch["input_ids"] != self.pad_text_token_id)
            batch["speech_attention_mask"] = (batch["speech_input_ids"] != self.pad_speech_token_id).max(dim=1)[0]

        local_max_seq_len = batch["input_ids"].shape[1]
        if local_max_seq_len % self.divider != 0:
            local_max_seq_len = self.divider * (local_max_seq_len // self.divider + 1)

        assert local_max_seq_len <= self.max_seq_len

        batch["pad_sizes"] = torch.tensor([batch["input_ids"].shape[1] - size for size in sizes])

        return batch


def build_tts_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
):
    assert (
        cfg.name == "tts"
    ), f"Tried to build image dataloader with cfg.name={cfg.name}"

    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset_hf.py"
        )

    streams_dict = cfg.dataset.pop("streams", None)
    mlm_probability = cfg.dataset.pop("mlm_probability", None)

    sft_mode = cfg.get("sft_mode", False)
    assert sft_mode, "Only sft_mode=True is supported for tts dataloader"

    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            streams.append(Stream(**stream))

    dataset = StreamingTTSDataset(
        tokenizer=tokenizer,
        streams=streams,
        cache_limit="256gb",
        sft_mode=sft_mode,
        **cfg.dataset,
    )

    collate_fn = PaddingCollatorWrapper(
        pad_text_token_id=cfg.dataset.text_modality.pad_token_id,
        pad_speech_token_id=cfg.dataset.speech_modality.pad_token_id,
        max_seq_len=cfg.dataset.max_seq_len,
        collate_attention_mask=cfg.get('collate_attention_mask', True)
    )

    if cfg.get("batch_type", None) == "flash_attn":
        raise NotImplementedError("Not implemented batch type 'flash_attn'")

    return DataLoader(
        dataset,
        drop_last=cfg.drop_last,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )
