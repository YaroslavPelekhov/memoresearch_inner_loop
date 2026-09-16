import logging
from typing import Optional

from llmfoundry.models.layers.attention import RING_ATTN_CLASSES

VALID_SP_SPLIT_TYPES = {"equal", "zigzag", "llama3"}


def get_sp_split_type(split_type: Optional[str] = None, attention_type: Optional[str] = None, varlen_input: bool = False):
    r"""Validate SP split type and return appropriate one based on provided attention type.

    If `split_type` is not provided, defaults to `zigzag` for RingAttn `attention_type`,
    `equal` otherwise.

    Raises error if non `zigzag` `split_type` is provided for RingAttn `attention_type`.
    """
    assert attention_type is not None, f"`attention_type` is not provided (got `None`)."

    if split_type is not None:
        if split_type not in VALID_SP_SPLIT_TYPES:
            raise ValueError(f"Got invalid `split_type` option! The only supported options are {VALID_SP_SPLIT_TYPES}, but got {split_type}.")

        if split_type in {"zigzag", "llama3"} and attention_type not in RING_ATTN_CLASSES:
            raise ValueError(
                (
                    "`zigzag` and `llama3` SP split types is only compatible with RingAttention classes "
                    f"(`LlamaPackedRingAttention`, `LlamaLatentRingAttention`), but got `{attention_type}` attention type."
                )
            )

        if split_type != "zigzag" and attention_type in RING_ATTN_CLASSES:
            logging.warning(f"Using non-zigzag split with RingAttn class {attention_type} may cause suboptimal performance.")
    else:
        if attention_type in RING_ATTN_CLASSES:
            if varlen_input:
                split_type = "llama3"
            else:
                split_type = "zigzag"
        else:
            split_type = "equal"

    return split_type
