import os
import json
import torch

from functools import partial
from omegaconf import DictConfig, OmegaConf
import torch.distributed
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any
from transformers import PreTrainedTokenizerBase
import hashlib
from datasets import Dataset
from composer import DataSpec
from composer.utils import dist
from llmfoundry.data.datasets.mmlu import format_few_shot_sample, format_subject
from llmfoundry.data.processors import GigaVisionProcessor
from composer.datasets.in_context_learning_evaluation import (
    InContextLearningMultipleChoiceTaskDataset,
    _tokenizer_needs_prefix_space,
    _make_padded_input,
    _get_continuation_span,
)

from llmfoundry.data.datasets.mmlu import _log_once

def _validate_sample_templates(label, prompt_string, prompt_query_template, prompt_choice_template):
    assert '{query}' in prompt_query_template, \
        'you need to pass "{query}" template'
    assert '{choice}' in prompt_choice_template, \
        'you need to pass "{choice_symbol}" and "{choice}" template'
    if label == "mmmu":
        assert '{subject}' in prompt_string, 'you need to pass "{subject}" template'


class VisionMCQADataset(InContextLearningMultipleChoiceTaskDataset):
    def __init__(
        self,
        label: str,
        dataset_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        pad_tok_id: int,
        num_fewshot: int,
        prompt_string: str,
        prompt_query_template: str,
        prompt_choice_template: str,
        prompt_answer_template: str,
        example_delimiter: str,
        continuation_delimiter: str,
        dataset_json_filename: str,
        dataset_images_dirname: str,
        few_shot_dataset_json_filename: Optional[str] = None,
        choice_num2symbol: Optional[Dict[int, str]] = None,
        sft_args: Optional[Dict[str, Any]] = None,
        sft_mode: bool = True,
        cached_datasets_directory_template: Optional[str] = None,
    ):
        assert dataset_json_filename.endswith('.jsonl'), 'only ".jsonl" format is supported'
        assert few_shot_dataset_json_filename is None or few_shot_dataset_json_filename.endswith('.jsonl'), 'only ".jsonl" format is supported'
        assert num_fewshot == 0, 'Fewshots are not fully implemented yet'

        self.choice_num2symbol = choice_num2symbol
        self.choice_symbol2num = {value: key for key, value in self.choice_num2symbol.items()}
        self.choice_symbols = list(map(lambda x: x[1], sorted(self.choice_num2symbol.items(), key=lambda x: x[0])))
        self.few_shot_dataset_json_filename = few_shot_dataset_json_filename
        self.label = label
        self.prompt_string = prompt_string
        self.prompt_query_template = prompt_query_template
        self.prompt_choice_template = prompt_choice_template
        self.prompt_answer_template = prompt_answer_template
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter

        _validate_sample_templates(label, prompt_string, prompt_query_template, prompt_choice_template)

        self.dataset_filepath = os.path.join(dataset_dir, dataset_json_filename)
        self.images_dirpath = os.path.join(dataset_dir, dataset_images_dirname)
        assert os.path.exists(self.dataset_filepath) and os.path.exists(self.images_dirpath), \
            (f'{self.dataset_filepath} or {self.images_dirpath} don\'t exist in dataset')

        self.prefix_space = _tokenizer_needs_prefix_space(tokenizer)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.num_fewshot = num_fewshot
        self.sft_mode = sft_mode
        self.sft_args = sft_args
        if sft_args is not None:
            for arg, value in sft_args.items():
                setattr(self, arg, value)

        self.processor_config.pop("name", default=None)
        self.processor = GigaVisionProcessor(**self.processor_config)

        self.list_keys = []
        self.static_keys = ['mode', 'generation_kwargs']
        self.tensor_keys = ['input_ids', 'images', 'labels', 'attention_mask']
        self.list_of_tensors_keys = ['continuation_indices']
        self.list_of_tuples_keys = ['choice_groupings']
        self.list_of_primitives = ['gold_indices']
        self.dataset_uri = dataset_json_filename
        self.cached_datasets_directory_template = cached_datasets_directory_template
        try:
            # if sft_args is OmegaConfig, convert it to dict
            self.sft_args = OmegaConf.to_container(sft_args)
        except ValueError:
            self.sft_args = sft_args
        # setting self.datasets
        self.load_cached_dataset()
        self.dataset.set_format(type='torch', columns=['images'], output_all_columns=True)
        self.num_choices = len(self.dataset[0]["choices"])

    def prep_examples(self, dataset_item):
        format_few_shot_partial = partial(
            format_few_shot_sample,
            prompt_query_template=self.prompt_query_template,
            prompt_choice_template=self.prompt_choice_template,
            prompt_answer_template=self.prompt_answer_template,
            continuation_delimiter=self.continuation_delimiter,
            example_delimiter=self.example_delimiter,
            choice_num2symbol=self.choice_num2symbol
        )

        dataset_item = self._extract_query_and_choices_content(dataset_item)
        dataset_item = self._tokenize_dataset_item(dataset_item, self.example_delimiter, format_few_shot_partial)
        return dataset_item
    def load_dataset(self):
        _log_once(f'Creating Vision dataset from scratch for {self.dataset_uri}...', "info")

        def gen():
            with open(self.dataset_filepath, mode="r") as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)
        # making writer_batch_size = 1, to exclude Overflow Exception.
        self.dataset = Dataset.from_generator(gen).map(self.prep_examples, writer_batch_size=1)
        return self.dataset

    def collate_fn(self, data):
        inputs = []
        images_list = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []

        max_num_images = max([dataset_item["images"].size(0) for dataset_item in data])

        tp_sp_group_size = dist.get_tp_sp_group_size()
        seq_len_divider = tp_sp_group_size if tp_sp_group_size is not None else 1

        for data_pair in data:
            choice_start_idx = len(continuation_indices)
            preamble, choices, gold_idx, images = (data_pair['preamble'], data_pair['choices'], data_pair['gold_idx'], data_pair['images'])

            if images.size(0) < max_num_images:
                padding_images = torch.zeros(
                    (max_num_images - images.size(0), images.size(1), images.size(2), images.size(3))
                )
                images = torch.cat((images, padding_images), dim=0)

            for choice in choices:
                inp = _make_padded_input(preamble, choice, self.max_seq_len,
                                         self.pad_tok_id)

                if inp.shape[0] % seq_len_divider !=0 :
                    pad_size = seq_len_divider - (inp.shape[0] % seq_len_divider)
                    inp = torch.nn.functional.pad(inp, pad=(0, pad_size), value=self.pad_tok_id)

                continuation_span = _get_continuation_span(preamble, choice)

                inputs.append(inp)
                continuation_indices.append(continuation_span)

                images_list.append(images)

            gold_idxs.append(gold_idx)
            choice_end_idx = len(continuation_indices)
            choice_groupings.append((choice_start_idx, choice_end_idx))

        # We run each distinct query + answer choice through the model separately and determine which
        # answer has the lowest per-token-perplexity.
        #
        # If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        # since the batch may consist of multiple questions, the choice_groupings indicates
        # which contiguous sequences of elements in the batch correspond to which question
        # gold_indices indicates which of the [0, N-1] choices is the correct one for each question.

        batch = {
            'input_ids': torch.stack(inputs),
            'images': torch.stack(images_list),
            'continuation_indices': continuation_indices,
            'mode': 'icl_task',
            'labels': torch.stack(inputs),
            'gold_indices': gold_idxs,
            'choice_groupings': choice_groupings,
        }
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch

    def _extract_query_and_choices_content(self, dataset_item: Dict[str, Any]):
        extracted_values = {
            "query": dataset_item["query"].strip(),
            "choices": [elem.strip() for elem in dataset_item["choices"]],
            "attachments": dataset_item["attachments"],
            "gold": self.choice_symbol2num[dataset_item["answer"]],
        }
        if dataset_item.get("subject"):
            extracted_values["subject"] = dataset_item["subject"]
        return extracted_values

    def _tokenize_dataset_item(self, dataset_item: Dict[str, Any], example_delimiter: str, format_few_shot_partial):
        prompt = self.prompt_string
        if self.label == "mmmu":
            prompt = self.prompt_string.format(subject=format_subject(dataset_item["subject"]))

        preamble = self.system_precursor + prompt + example_delimiter

        if self.label == "vim_mmmu":
            bare_question = ""
        else:
            bare_question = format_few_shot_partial(
                sample=dataset_item,
                include_answer=False
            )

        question = self._format_question_with_image_tokens(bare_question, len(dataset_item["attachments"]))

        assistant = self.assistant_precursor + self.sentinel_token
        if self.prefix_space:
            choice_symbols = [
                (f' {choice}' if not choice.startswith(' ') else choice)
                for choice in self.choice_symbols[:len(dataset_item["choices"])]
            ]
        else:
            choice_symbols = self.choice_symbols[:len(dataset_item["choices"])]
            assistant += ' '

        # adding tokens after system and before question
        sample = preamble + question + assistant
        sample_tokenized = self.tokenizer(sample, return_tensors="pt")["input_ids"][0]

        # remove eos token is exists
        if (
            self.tokenizer.eos_token_id is not None
            and len(sample_tokenized) > 1
            and sample_tokenized[-1] == self.tokenizer.eos_token_id
        ):
            sample_tokenized = sample_tokenized[:-1]

        choices_tokenized = [self.tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in choice_symbols]

        image_paths = [os.path.join(self.images_dirpath, attachment) for attachment in dataset_item["attachments"]]
        sample_tokenized, _, images_tensor = self.processor.preprocess(image_paths, sample_tokenized)

        return {
            "gold_idx": dataset_item["gold"],
            "preamble": sample_tokenized.tolist(),
            "choices": choices_tokenized,
            "images": images_tensor,
        }

    def get_self_args_string(self):
        """
        Returns a concise string representation of the relevant arguments.
        """
        processor_dict = self.processor.to_dict()
        relevant_args = {
            'dataset_json_filename': self.dataset_uri,
            'tokenizer': self.tokenizer.get_vocab(),
            'max_seq_len': self.max_seq_len,
            'pad_tok_id': self.pad_tok_id,
            'prompt_query_template': self.prompt_query_template,
            'prompt_choice_template': self.prompt_choice_template,
            'prompt_answer_template': self.prompt_answer_template,
            'dataset_filepath': self.dataset_filepath,
            'choice_num2symbol': str(self.choice_num2symbol),
            'num_fewshot': self.num_fewshot,
            "sft_mode": self.sft_mode,
            "sft_args" : self.sft_args,
            "images_dirpath" : self.images_dirpath,
            "few_shot_dataset_json_filename" : self.few_shot_dataset_json_filename

        }
        relevant_args.update(processor_dict)
        # convert the dictionary to a JSON string and hash it
        args_json = json.dumps(relevant_args, sort_keys=True)
        return hashlib.md5(args_json.encode()).hexdigest()

    def _format_question_with_image_tokens(self, bare_question: str, target_num_images: int):
        image_token_text = self.tokenizer.decode([self.processor.image_token_id]).strip()
        num_images_in_question = bare_question.count("[image_token]")
        if num_images_in_question > 0:
            # Since some tokenizers may have different representation for image_token,
            # we need to replace them
            bare_question = bare_question.replace("[image_token]", image_token_text)

        question = self.user_precursor
        if target_num_images > num_images_in_question:
            # We remove [image_tokens] from query to place them manually now.
            # Since some datasets like mmmu can have [image_tokens] inside choices that are not removed,
            # we need to add only missing `target_num_images - num_images_in_question` tokens
            question += " " + " ".join([image_token_text] * (target_num_images - num_images_in_question))

        question += " " + bare_question
        return question


def get_vision_mcqa_dataloader(
    cfg: DictConfig,
    batch_size: int,
    dataset_uri: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    pad_tok_id: int,
    num_fewshot: int,
    prompt_string: str,
    prompt_query_template: str,
    prompt_choice_template: str,
    prompt_answer_template: str,
    example_delimiter: str,
    continuation_delimiter: str,
    dataset_json_filename: str,
    dataset_images_dirname: str,
    few_shot_dataset_json_filename: Optional[str] = None,
    choice_num2symbol: Optional[Dict[int, str]] = None,
    cached_datasets_directory_template: str = None
):
    dataset = VisionMCQADataset(
        label=cfg.label,
        dataset_dir=dataset_uri,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_tok_id=pad_tok_id,
        num_fewshot=num_fewshot,
        prompt_string=prompt_string,
        prompt_query_template=prompt_query_template,
        prompt_choice_template=prompt_choice_template,
        prompt_answer_template=prompt_answer_template,
        example_delimiter=example_delimiter,
        continuation_delimiter=continuation_delimiter,
        dataset_json_filename=dataset_json_filename,
        dataset_images_dirname=dataset_images_dirname,
        few_shot_dataset_json_filename=few_shot_dataset_json_filename,
        choice_num2symbol=choice_num2symbol,
        sft_args=cfg.sft_args,
        cached_datasets_directory_template=cached_datasets_directory_template
    )

    batch_size = max(dataset.num_choices, batch_size)
    effective_batchsize = batch_size // dataset.num_choices

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)
    split_batch = dataset.split_batch

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=effective_batchsize,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            # # # # # # # #
            # i don't know why, but down presented variables
            # have been ignored in llm-foundry icl dataloaders
            num_workers=cfg.get('num_workers', 1),
            pin_memory=cfg.get('pin_memory', True),
            prefetch_factor=cfg.get('prefetch_factor', 2),
            persistent_workers=cfg.get('persistent_workers', False),
            # # # # # # # #
        ),
        device_transforms=None,
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=split_batch,
    )
