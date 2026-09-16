import torch
from composer.utils import dist

class TextCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int, scale: float = 1.0):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        tp_sp_group_size = dist.get_tp_sp_group_size()
        self.divider = tp_sp_group_size if tp_sp_group_size is not None else 1
        assert self.divider * scale == int(self.divider * scale), (
            "Devider is expected to be an int value but provided scale makes it non-int. "
            f"Scale={scale}, divider={self.divider}, divider * scale={self.divider * scale}."
        )
        self.divider = int(self.divider * scale)

    def __call__(self, text_examples):
        batch = {
            "input_ids": [],
            "labels": [],
        }

        max_seq_len = max(len(ex["input_ids"]) for ex in text_examples)

        if max_seq_len % self.divider != 0:
            max_seq_len = self.divider * (max_seq_len // self.divider + 1)

        assert max_seq_len <= self.max_seq_len, \
            f"Got sequence with max_seq_len={max_seq_len} which is larger than max_seq_len from config: {self.max_seq_len}"

        for ex in text_examples:
            seq_len = len(ex["input_ids"])
            pad_size = max_seq_len - seq_len

            batch["input_ids"].append(
                torch.nn.functional.pad(
                    ex["input_ids"],
                    (0, pad_size),
                    value=self.pad_token_id
                )
            )

            batch["labels"].append(
                torch.nn.functional.pad(
                    ex["labels"],
                    (0, pad_size),
                    value=-100
                )
            )

        out_dict = {k: torch.stack(v) for k, v in batch.items()}
        out_dict["local_max_seq_len"] = max_seq_len
        return out_dict
