# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import shutil
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Optional

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.data.data import check_dtype_optimality_and_safety

from .utils import validate_dataset


class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Convert dataset into MDS format, optionally concatenating and tokenizing"
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
    )
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default=None)
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument(
        "--tokens_dtype", type=str, default="uint16", required=True
    )

    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Validate dataset through iterating it before doing actual tokenization. Done to check if dataset can be loaded.",
    )

    parsed = parser.parse_args()

    if (
        os.path.isdir(parsed.out_root)
        and len(
            set(os.listdir(parsed.out_root)).intersection(set(parsed.split))
        )
        > 0
    ):
        raise ValueError(
            f"--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}."
        )

    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.tokenizer is None
    ):
        parser.error(
            "When setting --concat_tokens, you must specify a --tokenizer"
        )

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    return parsed


def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = "",
    eos_text: str = "",
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
    tokens_dtype: str = "uint16",
) -> IterableDataset:
    """Build an IterableDataset over provided JSONL dataset.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    if os.path.isdir(path):
        # Hugging Face preserves the supplied file-list order. Sort the glob so
        # a corpus rebuild is independent of concurrent download/creation order.
        data_files = sorted(glob(f"{path}/*"))
    else:
        data_files = path

    hf_dataset = hf_datasets.load_dataset(
        "json", data_files=data_files, split=split, streaming=True
    )

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f"{tokenizer=} must be of type PreTrainedTokenizerBase"
            )

        if max_length is None:
            raise ValueError(f"max_length must be set.")

        if bos_text + eos_text == "":
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id
            ):
                tok_error_msg = (
                    "This tokenizer does not insert an EOS nor BOS token. "
                )
                tok_error_msg += "Concatenating with this tokenizer will result in sequences being "
                tok_error_msg += "attached without a separating token. Please use another tokenizer, "
                tok_error_msg += "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)

        check_dtype_optimality_and_safety(tokenizer, tokens_dtype)
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            tokens_dtype=tokens_dtype,
        )

    return dataset


def generate_samples(
    loader: DataLoader, truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if (
                truncate_num_samples is not None
                and n_samples == truncate_num_samples
            ):
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        # Preserve LLM Foundry's documented pretraining conversion contract:
        # document boundaries come from --bos_text/--eos_text. Forcing the
        # tokenizer to add BOS/EOS here duplicates those explicit separators.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {"tokens": "bytes", "dtype": "int8"}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {"text": "str"}

    # Get samples
    dataset = build_hf_dataset(
        path=args.path,
        split=args.split,
        mode=mode,
        max_length=args.concat_tokens,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        no_wrap=args.no_wrap,
        tokenizer=tokenizer,
        tokens_dtype=args.tokens_dtype,
    )

    if args.validate_dataset:
        backup_base_path = None
        relative_cutoff = None
        assert backup_base_path is not None and relative_cutoff is not None, (
            "Set `backup_base_path` and `relative_cutoff` with proper values for bad shards backup!"
        )

        validate_dataset(
            dataset,
            source_path=args.path,
            backup_base_path=backup_base_path,
            relative_cutoff=relative_cutoff,
        )

    # Write samples
    print(f"Converting to MDS format...")
    print(
        f"Note that the progress bar is based on the dataset length before tokenization."
    )
    print(f"It will finish at a value below 100% if tokenizing")
    with MDSWriter(
        columns=columns,
        out=os.path.join(args.out_root),
        compression=args.compression,
    ) as out:
        for sample in dataset:
            out.write(sample)


if __name__ == "__main__":
    main(parse_args())
