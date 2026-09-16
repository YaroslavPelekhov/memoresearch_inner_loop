from typing import Mapping, Tuple, List, Optional, Dict, Any

from composer.utils import dist
from functools import reduce

from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss
import numpy as np

import torch
from torch import nn

from llmfoundry.models.parallel.sequence import gather_from_sequence_parallel_region

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.teacher_student import ComposerTeacherStudentModel

from composer.utils.dist import get_sp_group_size, get_sp_group, get_sp_group_rank

from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
)

from collections import UserDict


def filter_output(output: Any) -> torch.Tensor:
    """Extract logits from model output (handles HF outputs, dicts, tuples)."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        if output[0].ndim == 0:
            return output[1]
        return output[0]
    raise TypeError(f"Cannot extract logits from {type(output)}")


def build_mtp_weight_mask(
    mtp_mask: torch.tensor,
    mtp_loss_coef: float,
    device,
    dtype,
) -> torch.tensor:
    mtp_coef_mask = torch.ones_like(mtp_mask, device=device, dtype=dtype)

    if torch.any(mtp_mask):
        mtp_coef_mask[mtp_mask] = mtp_coef_mask[mtp_mask] * mtp_loss_coef

        main_elem_coef = mtp_mask.shape[0] / mtp_mask.to(dtype=dtype).sum()
        mtp_elem_coef = 1 / (1 - 1 / main_elem_coef)

        # NOTE: normalize loss to get main_loss.mean() + mtp_coef*coef_loss.mean()
        mtp_coef_mask[mtp_mask] = mtp_coef_mask[mtp_mask] * mtp_elem_coef
        mtp_coef_mask[~mtp_mask] = mtp_coef_mask[~mtp_mask] * main_elem_coef

    return mtp_coef_mask


def reshape_batch(
    batch: Mapping[str, torch.tensor],
    keys_to_reshape: List[str],
) -> Mapping[str, torch.tensor]:
    result = {}
    for key in batch:
        if key in keys_to_reshape:
            result[key] = batch[key].transpose(0, 1).reshape(-1, batch[key].size(-1))
        else:
            result[key] = batch[key]
    return result


def reshape_images(batch: Mapping[str, torch.Tensor]) -> None:
    """Reshape images tensor for chosen/rejected concatenation.

    Input images shape: [batch_size, 2*K, C, H, W] where first K are chosen, last K are rejected.
    Output images shape: [2*batch_size, K, C, H, W] with chosen first, rejected second.
    """
    im = batch["images"]
    K = im.size(1) // 2
    chosen_images = im[:, :K]
    rejected_images = im[:, K:]
    batch["images"] = torch.cat([chosen_images, rejected_images], dim=0)


def _build_shifted_labels(
    shift_labels: torch.Tensor,
    mtp_predictor_num: torch.Tensor,
    ignore_index: int = -100,
) -> List[torch.Tensor]:
    labels = [shift_labels]
    cur = shift_labels.clone()
    for _ in range(mtp_predictor_num):
        with torch.no_grad():
            cur = cur.roll(shifts=(-1,), dims=(1,)).contiguous()
            cur[:, -1] = (
                ignore_index  # NOTE: is it correct to mask full column but not only the last element of current row
            )
        labels.append(cur.clone())
    return torch.stack(labels, dim=0).view(-1, shift_labels.shape[-1])


def get_masked_sp_rank_labels(
    batch: Mapping[str, torch.tensor],
    mask_common_prefix: bool,
    masked_tokens: List[int],
    ignore_index: int = -100,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Function to convert input ids to labels and get token nums.
    Now this also handles splitting by SP ranks correctly (I hope)
    """
    action_mask = batch["action_mask"] * batch["attention_mask"]
    labels = batch["input_ids"]
    bs = labels.shape[0] // 2

    if mask_common_prefix:
        for row, index in enumerate((labels[:bs] != labels[bs:]).long().argmax(1)):
            action_mask[row, :index] = 0
            action_mask[row + bs, :index] = 0

    labels = torch.where(action_mask == 0, -100, labels)  # [:, 1:]

    if masked_tokens:
        masked_token_indexes = reduce(
            (lambda x, y: x | y), [labels == x for x in masked_tokens]
        )
        labels = torch.where(masked_token_indexes, -100, labels)
        action_mask_sum = (action_mask[:, 1:] * ~masked_token_indexes).sum(-1) + 1e-16
    else:
        action_mask_sum = action_mask[:, 1:].sum(-1) + 1e-16

    labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
    labels[:, -1] = ignore_index

    labels = get_sp_rank_labels(labels)

    return labels, action_mask_sum


def get_masked_labels(
    batch: Mapping[str, torch.tensor],
    mask_common_prefix: bool,
    masked_tokens: List[int],
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Function to convert input ids to labels and get token nums.
    """
    action_mask = batch["action_mask"] * batch["attention_mask"]
    labels = batch["input_ids"]
    bs = labels.shape[0] // 2

    if mask_common_prefix:
        for row, index in enumerate((labels[:bs] != labels[bs:]).long().argmax(1)):
            action_mask[row, :index] = 0
            action_mask[row + bs, :index] = 0

    labels = torch.where(action_mask == 0, -100, labels)[:, 1:]

    if masked_tokens:
        masked_token_indexes = reduce(
            (lambda x, y: x | y), [labels == x for x in masked_tokens]
        )
        labels = torch.where(masked_token_indexes, -100, labels)
        action_mask_sum = (action_mask[:, 1:] * ~masked_token_indexes).sum(-1) + 1e-16
    else:
        action_mask_sum = action_mask[:, 1:].sum(-1) + 1e-16

    return labels, action_mask_sum


def get_sp_rank_labels(labels: torch.Tensor) -> torch.Tensor:
    sp_rank = get_sp_group_rank() or 0
    sp_size = get_sp_group_size() or 1

    if sp_size == 1:
        return labels

    assert labels.shape[-1] % sp_size == 0, (
        f"You are trying to divide labels of length {labels.shape[-1]} into sp_size {sp_size} ranks. Expect problems to follow."
    )
    sp_rank_labels_len = labels.shape[-1] // sp_size

    sp_rank_labels = labels[
        ..., sp_rank * sp_rank_labels_len : (sp_rank + 1) * sp_rank_labels_len
    ].contiguous()

    return sp_rank_labels


def calculate_spectoken_nll(
    loss_fct: nn.Module,
    po_logits: torch.tensor,
    labels: torch.tensor,
    spectoken_list: List[int],
    mtp_mask: torch.tensor = None,
    mtp_loss_coef: float = 0,
    mtp_predictor_num: int | None = 0,
):
    if mtp_mask is None:
        mtp_mask = torch.zeros(po_logits.shape[0], device=po_logits.device)
        mtp_mask = mtp_mask == 1

    mtp_coef_mask = build_mtp_weight_mask(
        mtp_mask, mtp_loss_coef, po_logits.device, po_logits.dtype
    )

    num_sequences = mtp_predictor_num + 1
    batch_size = po_logits.shape[0] // 2 // num_sequences

    spectoken_indexes = reduce(
        (lambda x, y: x | y), [labels == x for x in spectoken_list]
    )
    spectoken_labels = torch.where(spectoken_indexes, labels, -100)

    po_logprobs_spectokens = -loss_fct(
        po_logits.reshape(-1, po_logits.size(-1)), spectoken_labels.reshape(-1)
    ).reshape(2 * batch_size * num_sequences, -1)

    sp_size = get_sp_group_size() or 1
    if sp_size > 1:
        sp_group = get_sp_group()
        spectoken_indexes_global = dist.all_gather(spectoken_indexes, group=sp_group)
        spectoken_indexes_global = torch.cat(spectoken_indexes_global, dim=-1)
        po_logprobs_spectokens_global = gather_from_sequence_parallel_region(
            po_logprobs_spectokens, tensor_parallel_output_grad=False
        )
    else:
        spectoken_indexes_global = spectoken_indexes
        po_logprobs_spectokens_global = po_logprobs_spectokens

    po_logprobs_spectokens_global = po_logprobs_spectokens_global.sum(-1)
    po_logprobs_spectokens_global = po_logprobs_spectokens_global * mtp_coef_mask

    chosen_po_nll_spectokens = -po_logprobs_spectokens_global[
        : batch_size * num_sequences
    ].mean() / (spectoken_indexes_global[: batch_size * num_sequences].sum(-1) + 1e-16)
    if torch.isnan(chosen_po_nll_spectokens).any():
        chosen_po_nll_spectokens = torch.tensor(0.0, requires_grad=True)
    return chosen_po_nll_spectokens


def calculate_chosen_and_rejected_logprobs(
    loss_fct: nn.Module,
    logits: torch.tensor,
    labels: torch.tensor,
    weighted_gamma: float = 1.0,
    n_tokens: int | None = None,
    mtp_predictor_num: int | None = 0,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Efficiently compute logprobs for chosen and rejected samples, supporting weighted gamma and token limits.
    """
    num_sequences = mtp_predictor_num + 1
    batch_size = logits.shape[0] // 2 // num_sequences

    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = labels.reshape(-1)

    nll = -loss_fct(logits_flat, labels_flat).reshape(
        2 * batch_size * num_sequences, -1
    )

    sp_size = get_sp_group_size() or 1
    if sp_size > 1:
        sp_group = get_sp_group()

        labels_global = dist.all_gather(labels, group=sp_group)
        labels_global = torch.cat(labels_global, dim=-1)
        nll_global = gather_from_sequence_parallel_region(
            nll, tensor_parallel_output_grad=False
        )  # NOTE: using gradient preserving gather
    else:
        labels_global = labels
        nll_global = nll

    mask = (labels_global != -100).int()
    cumulative = torch.cumsum(mask, dim=1)

    if n_tokens is not None and n_tokens != -1:
        token_mask = (cumulative <= n_tokens) & (mask == 1)
        nll_global = nll_global * token_mask

    if np.allclose(weighted_gamma, 1.0):
        logprobs = nll_global.sum(
            -1
        )  # NOTE: here we should use sum as then we calculate rewards for full sequence
        return logprobs[: batch_size * num_sequences], logprobs[
            batch_size * num_sequences :
        ]

    weights = torch.where(
        mask == 1, weighted_gamma ** (cumulative - 1), torch.zeros_like(mask)
    )
    assert weights.shape[0] == nll_global.shape[0], (
        "Weights are not global (probably split by SP rank)"
    )
    logprobs_weighted = (nll_global * weights).sum(
        -1
    )  # NOTE: here we should use sum as then we calculate rewards for full sequence

    return logprobs_weighted[: batch_size * num_sequences], logprobs_weighted[
        batch_size * num_sequences :
    ]


def calculate_nll(
    logprobs: torch.tensor,
    ref_logprobs: torch.tensor = None,
    norm: str = "length",
    token_num: torch.tensor = None,
    half_mtp_mask: torch.tensor = None,
    mtp_loss_coef: float = 0,
):
    if half_mtp_mask is None:
        half_mtp_mask = torch.zeros(logprobs.shape[0], device=logprobs.device)
        half_mtp_mask = half_mtp_mask == 1

    mtp_coef_mask = build_mtp_weight_mask(
        half_mtp_mask, mtp_loss_coef, logprobs.device, logprobs.dtype
    )

    if norm == "length":
        assert token_num is not None, "token_num is needed for length normalization"
        return (-logprobs / token_num) * mtp_coef_mask
    elif norm == "ref":
        assert ref_logprobs is not None, (
            "ref_logprobs is needed for length normalization"
        )
        return (logprobs / (ref_logprobs + 1e-10)) * mtp_coef_mask
    elif norm == "none":
        return -logprobs * mtp_coef_mask
    else:
        raise ValueError(f"Nll normalization doesn't support the type {norm}")


def calculate_po_loss(
    loss_type: str,
    chosen_po_logprobs: torch.tensor,
    rejected_po_logprobs: torch.tensor,
    beta_chosen: float,
    beta_rejected: float,
    gamma: float = 0,
    margin_coef: float = 0,
    margins: torch.tensor = None,
    chosen_ref_logprobs: torch.tensor = None,
    rejected_ref_logprobs: torch.tensor = None,
    chosen_tokens_num: torch.tensor = None,
    rejected_tokens_num: torch.tensor = None,
    half_mtp_mask: torch.tensor = None,
    mtp_loss_coef: float = 0,
) -> Tuple[torch.tensor, Optional[Dict[str, torch.Tensor]]]:
    metrics_dict = None
    if half_mtp_mask is None:
        half_mtp_mask = torch.zeros(
            chosen_po_logprobs.shape[0], device=chosen_po_logprobs.device
        )
        half_mtp_mask = half_mtp_mask == 1

    mtp_coef_mask = build_mtp_weight_mask(
        half_mtp_mask,
        mtp_loss_coef,
        chosen_po_logprobs.device,
        chosen_po_logprobs.dtype,
    )

    if loss_type == "dpo":
        chosen_rewards = beta_chosen * (chosen_po_logprobs - chosen_ref_logprobs)
        rejected_rewards = beta_rejected * (
            rejected_po_logprobs - rejected_ref_logprobs
        )
        reward_diff = chosen_rewards - rejected_rewards
        logits = reward_diff - margin_coef * margins
        po_loss = ((-nn.functional.logsigmoid(logits)) * mtp_coef_mask).mean()
        metrics_dict = {
            "chosen_reward": chosen_rewards.float().mean(),
            "rejected_rewards": rejected_rewards.float().mean(),
            "reward_diff": reward_diff.float().mean(),
            "dpo_diff": (chosen_po_logprobs - rejected_po_logprobs).float().mean(),
            "sft_diff": (chosen_ref_logprobs - rejected_ref_logprobs).float().mean(),
            "chosen_reward_mtp": chosen_rewards[half_mtp_mask].float().mean(),
            "rejected_rewards_mtp": rejected_rewards[half_mtp_mask].float().mean(),
        }

    elif loss_type == "simpo":
        reward = (
            beta_chosen * chosen_po_logprobs / chosen_tokens_num
            - beta_rejected * rejected_po_logprobs / rejected_tokens_num
            - gamma
        )
        po_loss = (-nn.functional.logsigmoid(reward) * mtp_coef_mask).mean()
    else:
        raise ValueError(f"loss_func={loss_type} is not implemented")
    return po_loss, metrics_dict


class ComposerDPOModel(ComposerTeacherStudentModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.beta_chosen = om_model_config.dpo.beta_chosen
        self.beta_rejected = om_model_config.dpo.beta_rejected
        self.gamma = om_model_config.dpo.gamma
        self.weighted_gamma = om_model_config.dpo.get("weighted_gamma", 1)
        self.n_tokens = om_model_config.dpo.get("n_tokens", None)
        self.margin_coef = om_model_config.dpo.margin_coef
        self.chosen_nll_coef = om_model_config.dpo.chosen_nll_coef
        self.loss_coef = om_model_config.dpo.loss_coef
        self.loss_func = om_model_config.dpo.loss_func
        self.spectokens_nll_coef = om_model_config.dpo.spectokens_nll_coef
        self.chosen_nll_norm = om_model_config.dpo.chosen_nll_norm
        self.mask_common_prefix = om_model_config.dpo.mask_common_prefix

        self.spectokens = om_model_config.dpo.spectokens
        self.masked_tokens = om_model_config.dpo.masked_tokens

        self.log_metrics = om_model_config.dpo.get("log_metrics", False)

        assert "teacher_model" in om_model_config, (
            "Teacher model is not defined in the config file"
        )

        if om_model_config.teacher_model.get("same_as_student", False):
            assert len(om_model_config.teacher_model.keys()) == 1, (
                "It is not allowed to set any parameters for teacher when `same_as_student=True`"
            )
            om_model_config.teacher_model = om_model_config.student_model

        if (
            hasattr(om_model_config.student_model.config_overrides, "use_mtp")
            and om_model_config.student_model.config_overrides.use_mtp is True
        ):
            self.use_mtp = True
            self.mtp_predictor_num = (
                om_model_config.student_model.config_overrides.mtp_predictor_num
            )
            self.mtp_loss_weight = (
                om_model_config.student_model.config_overrides.mtp_loss_weight
            )
        else:
            self.use_mtp = False
            self.mtp_predictor_num = 0
            self.mtp_loss_weight = 0

        train_metrics = []
        eval_metrics = [
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            om_model_config=om_model_config,
            tokenizer=tokenizer,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

    def forward(self, batch: Mapping):
        if not (isinstance(batch, dict) or isinstance(batch, UserDict)):
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model"
            )
        model_batch = {"input_ids": batch["input_ids"]}
        if "images" in batch:
            model_batch["images"] = batch["images"]
        if "attention_mask" in batch:
            model_batch["attention_mask"] = batch["attention_mask"]

        if not self.training:
            return self.model(**model_batch)
        else:
            model_batch = reshape_batch(model_batch, ["input_ids", "attention_mask"])
            if "images" in model_batch:
                reshape_images(model_batch)
            with torch.no_grad():
                teacher_out = self.teacher(**model_batch)
                teacher_out = filter_output(teacher_out)
                dist.barrier()

            student_out = self.model(**model_batch)
            student_out = filter_output(student_out)
            return (student_out, teacher_out)

    def get_loss_fct(self) -> nn.Module:
        loss_fct = FusedCrossEntropyLoss(
            reduction="none",
            inplace_backward=self.config.loss_inplace_backward,
            process_group=dist.get_tp_group() if self.training else None,
        )
        return loss_fct

    def loss(self, outputs, batch):
        """
        Compute the DPO loss and statistics for a batch of chosen and rejected dialogs.
        """

        metrics = None
        if self.log_metrics:
            metrics = {}

        batch = reshape_batch(batch, ["input_ids", "action_mask", "attention_mask"])
        bs = batch["input_ids"].shape[0] // 2
        last_chosen = bs

        po_logits, ref_logits = (
            outputs  # NOTE: (2*bs, seq_len, vocab_num)  mtp=0,  (2*bs*(mtp+1)*seq_len, vocab_size)  mtp>0
        )
        vocab_len = po_logits.shape[-1]

        real_seq_num = (
            (self.mtp_predictor_num + 1) * 2 * bs
        )  # NOTE: *2 because of chosen/rej sequences [cosen, rej, mtp_i_cosen, mtp_i_rej...]
        if self.use_mtp:
            output_real_len = po_logits.shape[0]

            po_logits = po_logits.view(
                self.mtp_predictor_num + 1,
                2,
                bs,
                output_real_len // real_seq_num,
                vocab_len,
            )
            po_logits = po_logits.permute(1, 2, 0, 3, 4)
            po_logits = po_logits.reshape(
                real_seq_num, output_real_len // real_seq_num, vocab_len
            )  # NOTE: ((mtp+1)*bs*2, seq_len, vocab_size)

            ref_logits = ref_logits.view(
                self.mtp_predictor_num + 1,
                2,
                bs,
                output_real_len // real_seq_num,
                vocab_len,
            )
            ref_logits = ref_logits.permute(1, 2, 0, 3, 4)
            ref_logits = ref_logits.reshape(
                real_seq_num, output_real_len // real_seq_num, vocab_len
            )

            # NOTE: mtp mask show wich sequencies are from mtp blocks
            # ----------------------------------------------------------------------------
            mtp_mask = torch.ones(
                (self.mtp_predictor_num + 1, 2, bs), device=po_logits.device
            )
            mtp_mask[0] = mtp_mask[0] * 0
            mtp_mask = mtp_mask.permute(1, 2, 0).reshape(real_seq_num)
            mtp_mask = mtp_mask == 1
            # ----------------------------------------------------------------------------

            last_chosen = last_chosen * (self.mtp_predictor_num + 1)
        else:
            mtp_mask = torch.zeros(real_seq_num, device=po_logits.device)
            mtp_mask = mtp_mask == 1
        loss_fct = self.get_loss_fct()

        # NOTE: include sp rank split inside
        labels, global_tokens_num = get_masked_sp_rank_labels(
            batch, self.mask_common_prefix, self.masked_tokens
        )

        if self.use_mtp:
            labels = _build_shifted_labels(
                labels, self.mtp_predictor_num, ignore_index=-100
            )

            labels = labels.view(
                self.mtp_predictor_num + 1, 2, bs, output_real_len // real_seq_num
            )
            labels = labels.permute(1, 2, 0, 3)
            labels = labels.reshape(real_seq_num, output_real_len // real_seq_num)

            global_tokens_num = global_tokens_num.repeat(
                self.mtp_predictor_num + 1
            ).unsqueeze(1)

            global_tokens_num = global_tokens_num.view(
                self.mtp_predictor_num + 1, 2, bs, 1
            )
            global_tokens_num = global_tokens_num.permute(1, 2, 0, 3)
            global_tokens_num = global_tokens_num.reshape(real_seq_num, 1)
        chosen_po_logprobs, rejected_po_logprobs = (
            calculate_chosen_and_rejected_logprobs(
                loss_fct,
                po_logits,
                labels,
                self.weighted_gamma,
                self.n_tokens,
                self.mtp_predictor_num,
            )
        )

        if self.loss_func == "dpo" or self.chosen_nll_norm == "ref":
            with torch.no_grad():
                chosen_ref_logprobs, rejected_ref_logprobs = (
                    calculate_chosen_and_rejected_logprobs(
                        loss_fct,
                        ref_logits,
                        labels,
                        self.weighted_gamma,
                        self.n_tokens,
                        self.mtp_predictor_num,
                    )
                )
                if self.log_metrics:
                    metrics.update(
                        {
                            "chosen_ref_logprobs": chosen_ref_logprobs.mean()
                            .detach()
                            .item(),
                            "rejected_ref_logprobs": rejected_ref_logprobs.mean()
                            .detach()
                            .item(),
                            "chosen_ref_logprobs_main": chosen_ref_logprobs[
                                ~mtp_mask[:last_chosen]
                            ]
                            .mean()
                            .detach()
                            .item(),
                            "rejected_ref_logprobs_main": rejected_ref_logprobs[
                                ~mtp_mask[:last_chosen]
                            ]
                            .mean()
                            .detach()
                            .item(),
                            "chosen_ref_logprobs_mtp": chosen_ref_logprobs[
                                mtp_mask[:last_chosen]
                            ]
                            .mean()
                            .detach()
                            .item(),
                            "rejected_ref_logprobs_mtp": rejected_ref_logprobs[
                                mtp_mask[:last_chosen]
                            ]
                            .mean()
                            .detach()
                            .item(),
                        }
                    )
        else:
            chosen_ref_logprobs, rejected_ref_logprobs = None, None

        po_loss, metrics_dict = calculate_po_loss(
            self.loss_func,
            chosen_po_logprobs,
            rejected_po_logprobs,
            self.beta_chosen,
            self.beta_rejected,
            self.gamma,
            self.margin_coef,
            batch["margins"],
            chosen_ref_logprobs,
            rejected_ref_logprobs,
            global_tokens_num[:last_chosen],
            global_tokens_num[last_chosen:],
            mtp_mask[: mtp_mask.shape[0] // 2],
            self.mtp_loss_weight,
        )

        if self.log_metrics:
            metrics_items = {k: v.detach().item() for k, v in metrics_dict.items()}
            metrics.update(metrics_items)

        if self.chosen_nll_coef:
            chosen_nll = calculate_nll(
                chosen_po_logprobs,
                chosen_ref_logprobs,
                self.chosen_nll_norm,
                global_tokens_num[:last_chosen],
                mtp_mask[: mtp_mask.shape[0] // 2],
                self.mtp_loss_weight,
            ).mean()
        else:
            chosen_nll = torch.tensor(0.0, requires_grad=True)

        # TODO: check corectness for calculate_spectoken_nll with new sp
        if self.spectokens_nll_coef:
            chosen_po_nll_spectokens = calculate_spectoken_nll(
                loss_fct,
                po_logits,
                labels,
                self.spectokens,
                mtp_mask,
                self.mtp_loss_weight,
                self.mtp_predictor_num,
            ).mean()
        else:
            chosen_po_nll_spectokens = torch.tensor(0.0, requires_grad=True)

        loss = (
            po_loss * self.loss_coef
            + chosen_nll * self.chosen_nll_coef
            + chosen_po_nll_spectokens * self.spectokens_nll_coef
        )

        if self.log_metrics:
            metrics.update(
                {
                    "po_loss": po_loss.detach().item(),
                    "chosen_nll": chosen_nll.detach().item(),
                    "chosen_po_nll_spectokens_loss": chosen_po_nll_spectokens.detach().item(),
                    "full_loss": loss.detach().item(),
                }
            )
            self.logger.log_metrics(metrics)

        return loss
