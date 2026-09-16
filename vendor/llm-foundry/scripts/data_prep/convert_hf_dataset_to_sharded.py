import os
from glob import glob
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

starcoder_langs = [
    "assembly",
    "awk",
    "c",
    "c-sharp",
    "cmake",
    "common-lisp",
    "cpp",
    "css",
    "cuda",
    "dockerfile",
    "go",
    "groovy",
    "html",
    "java",
    "javascript",
    "json",
    "kotlin",
    "lua",
    "makefile",
    "maple",
    "mathematica",
    "matlab",
    "powershell",
    "python",
    "r",
    "rust",
    "scala",
    "shell",
    "sql",
    "typescript",
    "yaml",
]

dataset_names = ["algebraic-stack", "open-web-math"]

# base_path = "/home/jovyalgebraic-stackan/isbaskov/clean_data/starcoderdata"
base_path = "/home/jovyan/isbaskov/clean_data/proof-pile-2"

# for lang in starcoder_langs:
for dataset_name in dataset_names:
    # Загрузка датасета
    # print(f'Loading {lang} subset...')
    # path = f'{base_path}/{lang}'  # starcoder
    path = f"{base_path}/{dataset_name}/train"  # other (lemma)
    data_files = glob(f"{path}/*")
    data_files = sorted(data_files)

    print("Loading datasets...")
    # dataset = Dataset.from_parquet(data_files, keep_in_memory=True, num_proc=16)

    # load lemma dataset one by one to evoid loading errors
    dataset_list = []
    for data_file in tqdm(data_files, desc="loading datasets"):
        dataset_list.append(
            Dataset.from_json(
                data_file,
                keep_in_memory=True,
            )
        )
    print(len(dataset_list))

    # remove meta column for lemma dataset to avoid errors and save space
    for idx in range(len(dataset_list)):
        dataset_list[idx] = dataset_list[idx].remove_columns("meta")
    dataset = concatenate_datasets(dataset_list)

    # change column name for starcoder dataset to match expected in convert scripts
    # dataset = dataset.rename_column(original_column_name="content", new_column_name="text")

    # Определение размера чанка
    chunk_size = 100_000  # Задайте свой размер чанка

    # Вычисление количества чанков
    num_chunks = len(dataset) // chunk_size + (1 if len(dataset) % chunk_size else 0)

    # Итерация по датасету с шагом размера чанка
    for shard_id in tqdm(range(num_chunks), desc=f"processing"):
        start_index = shard_id * chunk_size
        end_index = start_index + chunk_size

        # Берем часть от датасета
        chunk = dataset.shard(num_chunks, shard_id)

        # Задайте нужный путь и уникальное имя для каждого чанка
        chunk_path = f"/home/jovyan/datasets/raw/{dataset_name}-sharded/{shard_id:07}"
        assert not os.path.exists(
            chunk_path
        ), f"already exists, make new dir or empty this one: {chunk_path}"
        Path(chunk_path).mkdir(parents=True)

        # Сохранение чанка в файл
        chunk.save_to_disk(chunk_path)
