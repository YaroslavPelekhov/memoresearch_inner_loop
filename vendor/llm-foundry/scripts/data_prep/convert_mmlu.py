# Copyright 2023, SberDevices authors (v.mamedov)
from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

import datasets
import datasets as hf_datasets
from datasets import DownloadConfig

DATASET_NAME = 'cais/mmlu'
ANSWER2ID = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
RENAME_DICT = {
    'question': 'query',
    'subject': 'subject',
    'choices': 'choices',
    'answer': 'gold',
}

DESC = """
Script to reformat MMLU dataset to match with InContextLearningMultipleChoiceTaskDataset format.
    https://github.com/mosaicml/composer/blob/a1cd1fd18d6c80b1f64faa2213b39ddd61ddbcd0/composer/datasets/in_context_learning_evaluation.py#L444

Target format presented here:
    https://github.com/mosaicml/composer/blob/dev/tests/datasets/local_data/piqa_small.jsonl
"""


###################
# copy of llmfoundry.data.datasets.mmlu
# to avoid import problems
####################
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
        choice_num2symbol: Optional[dict[int, str]]=None,
        continuation_delimiter: str = '\n',
        example_delimiter: str = '\n\n',
        include_answer=True
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
    choice_num2symbol = choice_num2symbol or {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    _validate_sample_templates(prompt_query_template, prompt_choice_template)
    prompt = prompt_query_template.format(query=sample['query'])
    for i, choice in enumerate(sample['choices']):
        choice_string = prompt_choice_template.format(choice_symbol=choice_num2symbol[i], choice=choice)
        prompt += f"{continuation_delimiter}{choice_string}"
    prompt += f"{continuation_delimiter}{prompt_answer_template}"
    if include_answer:
        assert isinstance(sample['gold'], int)
        prompt += f" {choice_num2symbol[sample['gold']]}{example_delimiter}"
    return prompt


####################

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=DESC
    )

    parser.add_argument(
        '--hf-splits',
        nargs='+',
        default=[
            # do we need them?
            # 'auxiliary_train',
            # 'validation',
            'test',
            'dev'
        ])
    parser.add_argument(
        '--out-dir',
        type=lambda p: Path(p).absolute(),
        required=True
    )
    parser.add_argument(
        '--dataset-subset',
        type=str,
        default='all',
        required=False
    )
    parser.add_argument(
        '--num-proc',
        type=int,
        required=False,
        default=(os.cpu_count() // 2)
    )

    parsed = parser.parse_args()
    return parsed


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str


@dataclass
class DatasetConstants:
    hf_splits: dict[str, DataSplitConstants] = field(default_factory=dict)

    def add(self, dataset_consts: DataSplitConstants) -> None:
        self.hf_splits[dataset_consts.hf_split] = dataset_consts

    def __iter__(self):
        for _, v in self.hf_splits.items():
            yield v


mmlu4constants = DatasetConstants()
mmlu4constants.add(DataSplitConstants(hf_split='auxiliary_train', folder_split='train'))
mmlu4constants.add(DataSplitConstants(hf_split='dev', folder_split='few_shot'))
mmlu4constants.add(DataSplitConstants(hf_split='test', folder_split='test'))


def build_hf_dataset(
        dataset_name: str,
        dataset_subset: str,
        split_consts: DataSplitConstants,
        num_proc: int | None = None,
) -> datasets.Dataset:
    dataset = hf_datasets.load_dataset(
        path=dataset_name,
        name=dataset_subset,
        split=split_consts.hf_split,
        num_proc=num_proc,
        download_config=DownloadConfig(num_proc=num_proc)
    )
    return dataset


def main(args: Namespace) -> None:
    """
    Input:
        {
            "question": "What is the embryological origin of the hyoid bone?",
            "subject":"professional_accounting",
            "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches",
                        "The second pharyngeal arch", "The second and third pharyngeal arches"],
            "answer": "D"
        }
    Output:
        {
            "query": "What is the embryological origin of the hyoid bone?",
            "subject":"professional_accounting",
            "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches",
                        "The second pharyngeal arch", "The second and third pharyngeal arches"],
            "gold": "3"
        }
    Args:
        args (Namespace): Commandline arguments.
    """
    for hf_split_name in args.hf_splits:
        split_consts = mmlu4constants.hf_splits[hf_split_name]
        dataset = build_hf_dataset(
            dataset_name=DATASET_NAME,
            dataset_subset=args.dataset_subset,
            split_consts=split_consts,
            num_proc=args.num_proc
        )
        print(dataset[0])

        # remap with rename
        dataset = dataset.rename_columns(RENAME_DICT)
        dataset = dataset.align_labels_with_mapping(ANSWER2ID, 'gold')
        if split_consts.folder_split == 'train':
            print()
            format_few_shot_partial = partial(
                format_few_shot_sample,
                include_answer=True,
                prompt_query_template='{query}',
                prompt_choice_template='{choice_symbol}. {choice}',
                prompt_answer_template='Answer:',
                continuation_delimiter='\n',
                example_delimiter='\n\n',
                choice_num2symbol=None
            )
            dataset = dataset.map(
                lambda sample: {
                    "text": format_few_shot_partial(sample).strip()
                }
            )

        # Write samples
        # MDS writer doesn't support list of string
        #
        # with MDSWriter(
        #         columns=columns,
        #         out=args.out_dir / split_consts.folder_split,
        #         compression=args.compression
        # ) as out:
        #     for sample in tqdm(dataset, desc=split_consts.folder_split):
        #         out.write(sample)
        print(f'Converting {split_consts} to jsonl format...')
        out_dir = args.out_dir / args.dataset_subset
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_json(
            path_or_buf=out_dir / f"{split_consts.folder_split}.jsonl",
            num_proc=args.num_proc,
            orient='records',
            lines=True
        )


if __name__ == '__main__':
    main(parse_args())
