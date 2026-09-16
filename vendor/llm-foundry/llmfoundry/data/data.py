# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, Union, Optional, Any

import datasets as hf_datasets
import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, hf_dataset: Union[hf_datasets.IterableDataset,
                                         hf_datasets.Dataset]):
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # print(sample)
            # convert to bytes to store in MDS binary format
            yield {'text': sample['text'].encode('utf-8')}


DTYPE2ENCODING_MAPPING_DICT = {
    np.uint16: 0,
    np.uint32: 1,
    np.int64: 2,
    np.int32: 3,
}

ENCODING2DTYPE_MAPPING_DICT = {
    v: k for k, v in DTYPE2ENCODING_MAPPING_DICT.items()
}

def get_decoded_str_from_deserialized_bytes(deserialized_bytes):
        decoded_string = "".join([chr(code) for code in deserialized_bytes]).strip()
        return decoded_string

def read_binary_tokenized_sample(max_seq_len, sample: Dict[str, Any], field: str = 'tokens'):
        # take dtype or use np.int64
        sample_dtype_code = sample['dtype'] if 'dtype' in sample else 2
        assert sample_dtype_code in ENCODING2DTYPE_MAPPING_DICT, f"Use unsupported dtype code{sample_dtype_code}"
        tokens_dtype = ENCODING2DTYPE_MAPPING_DICT[sample_dtype_code]
        return torch.from_numpy(
            np.frombuffer(sample[field],
                          dtype=tokens_dtype)[:max_seq_len].copy().astype(np.int64))


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        tokens_dtype: Optional[str] = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        if tokens_dtype == "uint16":
            self.tokens_dtype = np.uint16
        elif tokens_dtype == "uint32":
            self.tokens_dtype = np.uint32
        elif tokens_dtype == "int32":
            self.tokens_dtype = np.int32
        else:
            self.tokens_dtype = np.int64
            tokens_dtype = "int64"

        if tokens_dtype == "int64":
            warnings.warn(
                """
                ================================================================================
                Use of int64 for tokenization is deprecated, please use uint16 or uint32 instead
                ================================================================================
                """
            )

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                +
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

        self._dry_iter = False

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        for sample in self.hf_dataset:
            if self._dry_iter:
                continue

            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample, dtype=self.tokens_dtype).tobytes(),
                    'dtype': DTYPE2ENCODING_MAPPING_DICT[self.tokens_dtype]
                }


def check_dtype_optimality_and_safety(tokenizer, tokens_dtype):
    MAX_TOKENS_FOR_2_BYTES = 65_536
    MAX_TOKENS_FOR_4_BYTES = 4_294_967_296

    if tokenizer.vocab_size <= MAX_TOKENS_FOR_2_BYTES and tokens_dtype != "uint16":
        warnings.warn("You should use uint16 for the tokenization as all of the possible ids in your vocab fit in uint16 type")
    elif MAX_TOKENS_FOR_2_BYTES < tokenizer.vocab_size <= MAX_TOKENS_FOR_4_BYTES and tokens_dtype not in ("uint32"):
        warnings.warn("You should use int32 for the tokenization as all of the possible ids in your vocab fit in uint32 type")

    if tokenizer.vocab_size > MAX_TOKENS_FOR_2_BYTES and tokens_dtype == "uint16":
        raise Exception(f"You must change tokens_dtype from uint16 as your vocab is bigger than {MAX_TOKENS_FOR_2_BYTES}")
    elif tokenizer.vocab_size > MAX_TOKENS_FOR_4_BYTES and tokens_dtype != "int64":
        raise Exception(f"You must change tokens_dtype to int64 as your vocab is bigger than {MAX_TOKENS_FOR_4_BYTES}")
