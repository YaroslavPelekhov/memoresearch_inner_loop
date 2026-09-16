from __future__ import annotations

import os
import os.path
import hashlib
from collections import defaultdict
from functools import partial
from typing import Union, Any, Dict, Optional
import json

import datasets
from datasets import Dataset
import torch
import transformers
from composer import DataSpec
from composer.datasets.in_context_learning_evaluation import InContextLearningMultipleChoiceTaskDataset, \
    _make_padded_input, strip_data, _tokenizer_needs_prefix_space, _get_continuation_span
from composer.utils import dist, get_file
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

from llmfoundry.utils.console_logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "MmluDataset",
    "DEFAULT_CHOICE_NUM2SYMBOL",
]

from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader


DEFAULT_CHOICE_NUM2SYMBOL = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
}


def _log_once(msg: str, mode: str = "info") -> None:
    if os.environ["RANK"] == "0":
        getattr(logger, mode)(msg)


def _validate_sample_templates(prompt_query_template, prompt_choice_template):
    assert '{query}' in prompt_query_template, \
        'you need to pass "{query}" template'
    assert '{choice_symbol}' in prompt_choice_template and '{choice}' in prompt_choice_template, \
        'you need to pass "{choice_symbol}" and "{choice}" template'


def format_subject(subject: str):
    """function from https://github.com/hendrycks/test/blob/master/evaluate_flan.py"""
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_few_shot_sample(
        sample: dict[str, Any],
        prompt_query_template: str,
        prompt_choice_template: str,
        prompt_answer_template: str,
        choice_num2symbol: dict[int, str],
        continuation_delimiter: str = '\n',
        example_delimiter: str = '\n\n',
        include_answer=True,
        babymmlu = False
):
    """
    Example args (asserts in main class):
        prompt_string = 'The following are multiple choice questions (with answers) about {subject}.'
        prompt_query_template = '{query}'
        prompt_choice_template = '{choice_symbol}. {choice}'
        prompt_answer_template = 'Answer:'

    Sample example:
        {
            "query": "What is the embryological origin of the hyoid bone?",
            "subject":"professional_accounting",
            "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches",
                        "The second pharyngeal arch", "The second and third pharyngeal arches"],
            "gold": "3"
        }
    """
    if not babymmlu:
        _validate_sample_templates(prompt_query_template, prompt_choice_template)
    prompt = prompt_query_template.format(query=sample['query'])
    if not babymmlu:
        for i, choice in enumerate(sample['choices']):
            choice_string = prompt_choice_template.format(choice_symbol=choice_num2symbol[i], choice=choice)
            prompt += f"{continuation_delimiter}{choice_string}"
    prompt += f"{continuation_delimiter}{prompt_answer_template}"
    if include_answer:
        assert isinstance(sample['gold'], int)
        if not babymmlu:
            prompt += f" {choice_num2symbol[sample['gold']]}{example_delimiter}"
        else:
            prompt += f" {sample['choices'][sample['gold']]}{example_delimiter}"
    return prompt


def pack_subjects_samples(few_shot_samples: datasets.Dataset) -> dict[str, list[dict]]:
    """
    Sample example:
        {
            "query": "What is the embryological origin of the hyoid bone?",
            "subject":"professional_accounting",
            "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches",
                        "The second pharyngeal arch", "The second and third pharyngeal arches"],
            "gold": "3"
        }
    """
    subject2samples: dict[str, list[dict]] = defaultdict(list)
    for sample in few_shot_samples:
        assert sample['subject'], 'empty subject can\'t be processed'
        subject2samples[sample['subject']] += [sample]
    if len(subject2samples) != 57:
        print('total num of subject is not like in MMLU.')
    # to avoid unexplicit get
    return dict(subject2samples)


class MmluDataset(InContextLearningMultipleChoiceTaskDataset):
    """
    This dataset overrides default behaviour of random few shot examples on fixed one.
    Also evaluating is equal original paper:
        https://github.com/hendrycks/test/blob/master/evaluate_flan.py

    ORIGINAL CLASS DESC:
    ---

    A dataset that construct batches for in-context learning multiple choice evaluation

    If each question has N answer choices, we construct N distinct inputs per question. In order to ensure
    consistency across multi-GPU, we set the batch size to be `min(N, batch_size)` so that all N
    inputs per question can stored in the same batch.

    Each batch then consists of batch_size // N distinct questions and has the following the structure

    'input_ids': Input tensor batch x seqlen x # tokens
    'continuation_indices': List of |batch| consisting of tensors indicating which indices in the sequence correspond to the question answer (aka continuation)
    'mode': Indicates to the model that this is an ICL task and may rely on a custom code path to properly update metrics
    'labels': Identical to the input, used by the model to calculate loss/metrics
    'gold_indices': List of length |batch_size // N| indicating for each question, which of the answers is correct (via an integer [0, N-1])
    'choice_groupings': Indicates which indices of the batch correspond to which questions

    ---
    Args:
        subject_dir (str): Either a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`. Dataset must consist of rows of JSON data points with "query",
            "choices", and "gold" index. See tests/datasets/local_data/piqa_small.jsonl.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used to transform data into batches
        max_seq_len (int): The sequence length expected by the model
        pad_tok_id (int): The special token reserved for padding the ends of batches
        destination_path (str): Temporary path to store downloaded datasets
        num_fewshot (int): The number of complete fewshot examples to prepend before each test example
        choice_num2symbol (dict[int, str] | None): map dict for map choice index num to symbol. Default mapping with 'ABCD'
        prompt_string (str): Prompt string to put once before all fewshot examples/test examples
        prompt_query_template (str): Prompt string to put sample's query
        prompt_choice_template (str): Prompt string to put sample's choice
        prompt_answer_template (str): Prompt string to put answer
        example_delimiter (str): Separator that goes between individual (context, continuation) pairs (e.g. '\n')
        continuation_delimiter: (str): Separator that goes between context and continuation in each example (e.g. '->')
        dataset_json_filename: (str): evaluating file name
        few_shot_dataset_json_filename: (str):
    """

    # !!!
    #  Arguments are passing in llmfoundry/utils/builders.py
    #  so set defaults there
    # !!!
    def __init__(
            self,
            subject_dir: str,
            tokenizer: Union[transformers.PreTrainedTokenizerBase, transformers.PreTrainedTokenizerFastBase],
            max_seq_len: int,
            pad_tok_id: int,
            num_fewshot: int,
            prompt_string: str,  # = 'The following are multiple choice questions (with answers) about {subject}.',
            prompt_query_template: str,  # = '{query}',
            prompt_choice_template: str,  # = '{choice_symbol}. {choice}',
            prompt_answer_template: str,  # = 'Answer:',
            example_delimiter: str,  # = '\n\n',
            continuation_delimiter: str,  # = '\n',
            dataset_json_filename: str,  # = 'test.jsonl',
            few_shot_dataset_json_filename: str,  # = 'few_shot.jsonl',
            choice_num2symbol: dict[int, str] | None,  # = None,
            cached_datasets_directory_template: Optional[str] = None,
            babymmlu: bool = False,
            sft_mode: bool = False,
            sft_args: Dict | None = None,
            *args
    ):
        assert dataset_json_filename.endswith('.jsonl'), 'only ".jsonl" format is supported'
        assert few_shot_dataset_json_filename.endswith('.jsonl'), 'only ".jsonl" format is supported'
        assert num_fewshot >= 0, 'fewshot num could be only non-negative'

        self.choice_num2symbol = choice_num2symbol or DEFAULT_CHOICE_NUM2SYMBOL
        self.choice_symbols = list(map(lambda x: x[1], sorted(self.choice_num2symbol.items(), key=lambda x: x[0])))
        # prompt
        self.num_fewshot = num_fewshot
        self.prompt_string = prompt_string
        self.example_delimiter = example_delimiter
        self.continuation_delimiter = continuation_delimiter
        self.prompt_query_template = prompt_query_template
        self.prompt_choice_template = prompt_choice_template
        self.prompt_answer_template = prompt_answer_template
        # prompt's asserts
        if not babymmlu:
            assert '{subject}' in prompt_string, \
                print('you need to pass "{subject}" template')

        if not babymmlu:
            _validate_sample_templates(prompt_query_template, prompt_choice_template)
        #
        self.dataset_filepath = os.path.join(subject_dir, dataset_json_filename)
        self.dataset_uri = subject_dir # dataset uri isn't needed for MMLU, but is needed for proper logging
        self.subject_name = os.path.dirname(subject_dir)
        self.few_shot_filepath = os.path.join(subject_dir, few_shot_dataset_json_filename)
        assert os.path.exists(self.dataset_filepath) and os.path.exists(self.few_shot_filepath), \
            (f'{dataset_json_filename} or {few_shot_dataset_json_filename} don\'t exist in '
             f'subject: {self.subject_name}')

        disable_progress_bar()


        # few shot
        # changed from load_dataset to this implementation for throughput reasons on some clusters
        few_shot_dataset = Dataset.from_generator(self.jsonl_generator, gen_kwargs={"filepath": self.few_shot_filepath})
        ## preprocess
        self.subject2few_shot_samples = pack_subjects_samples(few_shot_dataset)

        # changed from load_dataset to this implementation for throughput reasons on some clusters
        dataset = Dataset.from_generator(self.jsonl_generator, gen_kwargs={"filepath": self.dataset_filepath})
        self.samples = list(
            dataset.map(lambda examples: {
                'query': examples['query'],
                'choices': examples['choices'],
                'gold': examples['gold']
            }))
        self.samples = [strip_data(entry) for entry in self.samples]

        self.num_choices = len(self.samples[0]['choices']) # note that for MMLU-PRO it is meaningless
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_tok_id = pad_tok_id
        self.sft_mode = sft_mode
        try:
            # if sft_args is OmegaConfig, convert it to dict
            self.sft_args = OmegaConf.to_container(sft_args)
        except ValueError:
            self.sft_args = sft_args

        if sft_args is not None:
            for arg, value in sft_args.items():
                setattr(self, arg, value)

        self.prefix_space = _tokenizer_needs_prefix_space(self.tokenizer)

        self.babymmlu = babymmlu

        self.list_keys = []
        self.static_keys = ['mode', 'generation_kwargs']
        self.tensor_keys = ['input_ids', 'labels', 'attention_mask']
        self.list_of_tensors_keys = ['continuation_indices']
        self.list_of_tuples_keys = ['choice_groupings']
        self.list_of_primitives = ['gold_indices']

        self.cached_datasets_directory_template = cached_datasets_directory_template

        self.load_cached_dataset()

    def jsonl_generator(self, filepath):
        """
        Reads JSONL file line-by-line.
        On some clusters this is actually faster than doing load_dataset() from the same file
        (although the reason for that is not clear at the moment)
        """
        with open(filepath, "r") as f:
            for line in f:
                yield json.loads(line)

    def load_dataset(self):
        _log_once(f'Creating MMLU-like dataset from scratch for {self.dataset_uri}...', "info")
        self.dataset = self.prep_examples(self.num_fewshot, self.prompt_string, self.example_delimiter, self.continuation_delimiter)

        return self.dataset

    def get_self_args_string(self):
        """
        Returns a concise string representation of the relevant arguments.
        """
        relevant_args = {
            'subject_name': self.subject_name,
            'tokenizer': self.tokenizer.get_vocab(),
            'max_seq_len': self.max_seq_len,
            'pad_tok_id': self.pad_tok_id,
            'prompt_query_template': self.prompt_query_template,
            'prompt_choice_template': self.prompt_choice_template,
            'prompt_answer_template': self.prompt_answer_template,
            'dataset_filepath': self.dataset_filepath,
            'few_shot_filepath': self.few_shot_filepath,
            'choice_num2symbol': str(self.choice_num2symbol),
            'num_fewshot': self.num_fewshot,
            'babymmlu': self.babymmlu,
            'sft_mode': self.sft_mode,
            'sft_args': self.sft_args,
        }
        # convert the dictionary to a JSON string and hash it
        args_json = json.dumps(relevant_args, sort_keys=True)
        return hashlib.md5(args_json.encode()).hexdigest()

    def prep_examples(
            self,
            num_fewshot: int,
            prompt_string: str,
            example_delimiter: str,
            continuation_delimiter: str,
            *args
    ):
        """
            Making examples like in original paper.
            https://github.com/hendrycks/test/blob/master/evaluate_flan.py#L32
        """
        examples = []

        format_few_shot_partial = partial(
            format_few_shot_sample,
            prompt_query_template=self.prompt_query_template,
            prompt_choice_template=self.prompt_choice_template,
            prompt_answer_template=self.prompt_answer_template,
            continuation_delimiter=continuation_delimiter,
            example_delimiter=example_delimiter,
            choice_num2symbol=self.choice_num2symbol,
            babymmlu=self.babymmlu
        )
        for sample in self.samples:
            subject = sample['subject']
            preamble = prompt_string
            preamble = preamble.format(subject=format_subject(subject))
            if self.sft_mode:
                preamble = self.system_precursor + preamble
            preamble += example_delimiter

            # same thing like in original repo
            #  to avoid context len overflow
            k = num_fewshot
            _preamble, _question, _assisnant = None, None, None
            while k > -1:
                few_shot_prompt = ""
                fewshot_samples: list[dict] = self.subject2few_shot_samples[subject]
                for fewshot_sample in fewshot_samples[:k]:
                    few_shot_prompt += format_few_shot_partial(
                        sample=fewshot_sample,
                        include_answer=True
                    )

                if self.sft_mode:
                    _preamble = preamble + few_shot_prompt
                    _question = self.user_precursor + format_few_shot_partial(
                        sample=sample,
                        include_answer=False
                    )
                    _assisnant = self.assistant_precursor + self.sentinel_token

                    _preamble_tokenized = self.tokenizer(_preamble)['input_ids']
                    _question_tokenized = self.tokenizer(_question)['input_ids']
                    _assisnant_tokenized = self.tokenizer(_assisnant)['input_ids']

                    _preamble += _question + _assisnant
                    _input_tokens = _preamble_tokenized + _question_tokenized + _assisnant_tokenized
                else:
                    few_shot_prompt += format_few_shot_partial(
                        sample=sample,
                        include_answer=False
                    )
                    _preamble = preamble + few_shot_prompt
                    _input_tokens = self.tokenizer(_preamble)['input_ids']

                if len(_input_tokens) > self.max_seq_len:
                    logger.info(f"{self.max_seq_len} is not enough, met {len(_input_tokens)}")
                    k -= 1
                else:
                    preamble = _preamble
                    break
            encoded_example = {}
            if self.prefix_space:
                choice_symbols = [
                    (f' {choice}' if not choice.startswith(' ') and not self.babymmlu else choice)
                    for choice in self.choice_symbols
                ]
            else:
                choice_symbols = self.choice_symbols
                if not self.babymmlu:
                    preamble += ' '


            if self.babymmlu:
                choice_symbols = sample['choices']

            if self.sft_mode:
                _preamble_tokenized = self.tokenizer(_preamble)
                _question_tokenized = self.tokenizer(_question)
                _assisnant_tokenized = self.tokenizer(_assisnant)
                for k in _preamble_tokenized.keys():
                    _preamble_tokenized[k] = _preamble_tokenized[k] + _question_tokenized[k] + _assisnant_tokenized[k]
                encoded_example['preamble'] = _preamble_tokenized
            else:
                encoded_example['preamble'] = self.tokenizer(preamble)


            # remove eos token is exists
            if (
                    self.tokenizer.eos_token_id is not None
                    and len(encoded_example['preamble']['input_ids']) > 1
                    and encoded_example['preamble']['input_ids'][-1] == self.tokenizer.eos_token_id
            ):
                encoded_example['preamble']['input_ids'] = encoded_example['preamble']['input_ids'][:-1]

            encoded_example['gold_idx'] = sample['gold']
            encoded_example['choices'] = [self.tokenizer(choice, add_special_tokens=False) for choice in choice_symbols]

            examples.append(encoded_example)

        return Dataset.from_list(examples)

    def collate_fn(self, data):
        """
        almost same as in original class,
            but "context" field removed from data
        """
        inputs = []
        continuation_indices = []
        gold_idxs = []
        choice_groupings = []
        max_seq_len_in_batch = -1
        for data_pair in data:

            choice_start_idx = len(continuation_indices)
            preamble, choices, gold_idx = (data_pair['preamble'], data_pair['choices'], data_pair['gold_idx'])

            for choice in choices:
                context_enc = preamble['input_ids']
                continuation_enc = choice['input_ids']
                max_seq_len_in_batch = max(len(context_enc), max_seq_len_in_batch)

                continuation_span = _get_continuation_span(context_enc, continuation_enc)
                inp = _make_padded_input(context_enc, continuation_enc, self.max_seq_len,
                                         self.pad_tok_id)
                inputs.append(inp)
                continuation_indices.append(continuation_span)

            gold_idxs.append(gold_idx)
            choice_end_idx = len(continuation_indices)
            choice_groupings.append((choice_start_idx, choice_end_idx))
        # little optimization, other icl dataloader in progress
        # max_seq_len_in_batch = min(max_seq_len_in_batch+1, self.max_seq_len)
        # We run each distinct query + answer choice through the model separately and determine which
        # answer has the lowest per-token-perplexity.
        #
        # If each question has N possible choices, all N must be grouped together as distinct elements of the batch
        # since the batch may consist of multiple questions, the choice_groupings indicates
        # which contiguous sequences of elements in the batch correspond to which question
        # gold_indices indicates which of the [0, N-1] choices is the correct one for each question.
        batch = {
            'input_ids': torch.stack(inputs), #[:, :max_seq_len_in_batch],
            'continuation_indices': continuation_indices,
            'mode': 'icl_task',
            'labels': torch.stack(inputs), #[:, :max_seq_len_in_batch],
            'gold_indices': gold_idxs,
            'choice_groupings': choice_groupings
        }
        batch['attention_mask'] = ~(batch['input_ids'] == self.pad_tok_id)
        return batch


def build_mmlu_dataloader(
        cfg: DictConfig,
        subject_dir: str,
        batch_size: int,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
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
        few_shot_dataset_json_filename: str,
        choice_num2symbol: dict[int, str],
        cached_datasets_directory_template: Optional[str] = None,
        babymmlu: bool = False,
        sft_mode: bool = False,
        sft_args: Dict | None = None,

) -> DataSpec:
    dataset = MmluDataset(
        subject_dir=subject_dir,
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
        few_shot_dataset_json_filename=few_shot_dataset_json_filename,
        choice_num2symbol=choice_num2symbol,
        cached_datasets_directory_template=cached_datasets_directory_template,
        babymmlu=babymmlu,
        sft_mode=sft_mode,
        sft_args=sft_args,
    )
    # ## Remove effective_batchsize because we change logic for microbatching in Composer _eval_loop method
    # effective_batchsize = batch_size

    ## effective batch_size
    batch_size = max(dataset.num_choices, batch_size)
    effective_batchsize = batch_size // dataset.num_choices

    _log_once(f'Set effective_batchsize = {effective_batchsize}, num_coices={dataset.num_choices}, batch_size={batch_size} for dataset {dataset_json_filename}...', "info")

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
            prefetch_factor=cfg.get('prefetch_factor', None),
            persistent_workers=cfg.get('persistent_workers', False),
            # # # # # # # #
        ),
        device_transforms=None,
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=split_batch,
    )


def get_mmlu_dataloader(
        cfg: DictConfig,
        batch_size: int,  # The size of a batch used for evaluation
        dataset_uri: str,  # The URI of the dataset
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        # The tokenizer used for data transformation
        max_seq_len: int,  # The maximum sequence length expected by the model
        pad_tok_id: int,  # The token ID for padding
        num_fewshot: int,  # The number of fewshot examples to pad each test example
        prompt_string: str,  # The prompt string to put before all fewshot examples/test examples
        prompt_query_template: str,  # The template for the query prompt
        prompt_choice_template: str,  # The template for the choice prompt
        prompt_answer_template: str,  # The template for the answer prompt
        example_delimiter: str,  # The delimiter between individual examples
        continuation_delimiter: str,  # The delimiter between context and continuation in each example
        dataset_json_filename: str,  # The filename for the dataset JSON file
        few_shot_dataset_json_filename: str,  # The filename for the fewshot dataset JSON file
        choice_num2symbol: dict[int, str] | None,  # A mapping from choice number to symbol (optional)
        has_categories: bool = False,  # Whether the dataset has categories
        cached_datasets_directory_template: Optional[str] = None,
        babymmlu: bool = False
) -> Union[DataSpec, Dict[str, DataSpec]]:
    """
    This constructs a dataloader (or dataloaders if has_categories is True) capable of evaluating LLMs on in-context
    learning language modeling tasks.

    Args:
        batch_size (int): The size of a batch used for evaluation
        dataset_uri (str): The URI of the dataset
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): The tokenizer used for data transformation
        max_seq_len (int): The maximum sequence length expected by the model
        pad_tok_id (int): The token ID for padding
        num_fewshot (int): The number of fewshot examples to pad each test example
        prompt_string (str): The prompt string to put before all fewshot examples/test examples
        prompt_query_template (str): The template for the query prompt
        prompt_choice_template (str): The template for the choice prompt
        prompt_answer_template (str): The template for the answer prompt
        example_delimiter (str): The delimiter between individual examples
        continuation_delimiter (str): The delimiter between context and continuation in each example
        dataset_json_filename (str): The filename for the dataset JSON file
        few_shot_dataset_json_filename (str): The filename for the fewshot dataset JSON file
        choice_num2symbol (dict[int, str] | None): A mapping from choice number to symbol (optional)
        has_categories (bool): Whether the dataset has categories

    Returns:
        Union[DataSpec, Dict[str, DataSpec]]: The dataloader or dataloaders used for performing in-context learning evaluation on the dataset provided

    ---
    example config:

    ```yaml
    icl_tasks:
      - label: mmlu
        dataset_uri: /home/jovyan/vmamedov/mmlu/all
        # each num_fewshot is separate dataloader
        num_fewshot: [ 3, 5 ]
        icl_task_type: mmlu
        ####################
        # only for mmlu
        # be careful, because the settings
        # applied to all dataloaders in this tasks
        drop_last: false
        num_workers: 4
        pin_memory: true
        prefetch_factor: 2
        persistent_workers: false
        timeout: 0
        ####################
    ```
    """
    dataloaders_pretrain = build_mmlu_dataloader(
        cfg,
        dataset_uri,
        batch_size,
        tokenizer,
        max_seq_len,
        pad_tok_id,
        num_fewshot,
        prompt_string,
        prompt_query_template,
        prompt_choice_template,
        prompt_answer_template,
        example_delimiter,
        continuation_delimiter,
        dataset_json_filename,
        few_shot_dataset_json_filename,
        choice_num2symbol,
        cached_datasets_directory_template,
        babymmlu=babymmlu,
        sft_mode=False,
        sft_args=None,
    )
    if has_categories:
        # raise NotImplementedError('categories not yet supported')
        dataloaders_sft = build_mmlu_dataloader(
            cfg,
            dataset_uri,
            batch_size,
            tokenizer,
            max_seq_len,
            pad_tok_id,
            num_fewshot,
            prompt_string,
            prompt_query_template,
            prompt_choice_template,
            prompt_answer_template,
            example_delimiter,
            continuation_delimiter,
            dataset_json_filename,
            few_shot_dataset_json_filename,
            choice_num2symbol,
            cached_datasets_directory_template,
            babymmlu=babymmlu,
            sft_mode=True,
            sft_args=cfg.sft_args,
        )
        return {
            'pretrain': dataloaders_pretrain,
            'sft': dataloaders_sft,
        }
    return dataloaders_pretrain
