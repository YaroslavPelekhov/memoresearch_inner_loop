import json
import os
import shutil
from pathlib import Path

from pyarrow.lib import ArrowInvalid

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset

IGNORE_FILES = ["README.md", ".gitattributes"]


def prepare_data_splits(
    input_path: str,
    output_path: str,
    task_name: str,
    num_executors: int = 1,
    per_file: bool = False,
):
    """
    Prepare data splits based on the provided `num_executors`. Splits input and output path pairs
    into chunks that each executor can process independently and saves to a JSON file as a list of lists
    using provided `task_name`.

    Parameters
    ----------
    input_path: str
        Path to a root folder of original data that needs processing. Root folder and all the subfolders
        will be added to a processing list.
    output_path: str
        Path to an output folder where the processed data will be saved.
    task_name: str
        Name of the JSON task file with input-output processing path pairs splits for each executor.
    num_executors: int, default=1
        Number of executors to use for data processing. Each executor processes each own data split.
        Defaults ot ``1``.
    per_file: bool, default=False
        Whether to perform data split per file or per folders, containing data files.
        Defaults to ``False``.

    Returns
    -------
    str:
        Path to saved task file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Provided input path does not exist: {input_path}!"
        )

    assert os.path.isdir(input_path), (
        f"Input path must be dir, provided {input_path}!"
    )
    source_paths = set()
    for root, dirs, files in os.walk(input_path):
        if len(dirs) > 0 and len(files) > 0:
            if not per_file:
                error_msg = (
                    f"There are some files and dirs in the root directory {root}, but "
                    "it supposed to have either only files or only dirs. Please, check it up."
                )
                raise RuntimeError(error_msg)

        if "ipynb_checkpoints" in root:
            print(
                f"Met .ipynb_checkpoints directory in {root} while traversing the file tree, skipping it."
            )
            continue

        if len(files) > 0:
            if per_file:
                for file in files:
                    if file in IGNORE_FILES:
                        continue
                    source_paths.add(os.path.join(root, file))
            else:
                source_paths.add(root)

    source_paths = tuple(sorted(source_paths))
    dest_paths = tuple(
        path.replace(input_path, output_path) for path in source_paths
    )

    task_data = []
    for i in range(len(source_paths)):
        task_data.append((source_paths[i], dest_paths[i]))

    exucutors_partitions = []
    partition_size = len(task_data) // num_executors + 1
    for i in range(num_executors):
        partition = task_data[i * partition_size : (i + 1) * partition_size]
        exucutors_partitions.append(partition)

    tasks_folder = Path(__file__).parent / "tasks"
    if not tasks_folder.exists():
        os.makedirs(tasks_folder, exist_ok=True)
    task_file_path = str(tasks_folder / f"{task_name}.json")
    with open(task_file_path, "w") as fout:
        json.dump(exucutors_partitions, fout)

    return task_file_path


def validate_config(config: dict):
    """Validate tokenization config and its options. If something is wrong, the error is raised.

    Parameters
    ----------
    config: dict
        Config dict to validate.
    """
    if config["base_image"] is None:
        raise RuntimeError("Base run image is not provided.")

    if not os.path.exists(config["base_input_path"]):
        raise RuntimeError(
            f"Provided base input path does not exist: {config['base_input_path']}."
        )

    if not os.path.exists(config["tokenizer_path"]):
        raise RuntimeError(
            f"Provided tokenizer does not exist: {config['tokenizer_path']}."
        )

    if config["max_tokenized_len"] is None or config["max_tokenized_len"] < 0:
        raise RuntimeError("`max_tokenized_len` is not set or <= 0.")

    if "n_proc" not in config or config["n_proc"] < 1:
        raise RuntimeError(
            "Config is expected to have `n_proc` field that is >= 0."
        )

    if "n_gpus" not in config or config["n_gpus"] < 1 or config["n_gpus"] > 8:
        raise RuntimeError(
            "Config is expected to have `n_gpus` field that is >= 0 and <= 8."
        )

    for dataset in config["datasets"]:
        dataset_config = config["datasets"][dataset]
        if (
            "num_executors" not in dataset_config
            or dataset_config["num_executors"] < 1
        ):
            raise RuntimeError(
                f"No `num_executors` provided for dataset {dataset} or it is less than 1."
            )

        if "per_file_task_split" not in dataset_config:
            raise RuntimeError(
                f"No `per_file_task_split` provided for dataset {dataset}."
            )

        if "script_path" not in dataset_config or not os.path.exists(
            dataset_config["script_path"]
        ):
            raise RuntimeError(
                "Tokenization script path `script_path` field is not present in dataset config or it does not exist."
            )

        dataset_path = os.path.join(config["base_input_path"], dataset)
        if not os.path.exists(dataset_path):
            raise RuntimeError(
                f"Provided dataset does not exist: {dataset_path}."
            )


def load_config(path: str) -> dict:
    """Load tokenization config. Override some options if necessary.

    Parameters
    ----------
    path: str
        Path to a config in a JSON format.

    Returns
    -------
    dict:
        Read and processed config in dict form.
    """
    with open(path) as fin:
        config = json.load(fin)

    validate_config(config)

    return config


def is_merged(path: str) -> bool:
    if os.path.exists(path) and os.path.isdir(path):
        dataset_files = os.listdir(path)
        contains_dirs = any(
            os.path.isdir(os.path.join(path, file)) for file in dataset_files
        )
        if "index.json" in dataset_files and not contains_dirs:
            return True

    return False


def validate_dataset(
    dataset: ConcatTokensDataset | NoConcatDataset,
    source_path: str | None = None,
    backup_base_path: str | None = None,
    relative_cutoff: int | None = None,
) -> None:
    # Iter through all the samples to check if they are okay.
    # This is to fix an error when code samples are broken to go through and cause
    # an error.
    if not isinstance(dataset, ConcatTokensDataset):
        raise ValueError(
            f"Provided dataset for validation is expected to be ConcatTokensDataset, but got {type(dataset).__name__}."
        )

    backup_bad_data = False
    if source_path is not None and os.path.exists(source_path):
        source_path = os.path.abspath(source_path)
        if backup_base_path is None:
            backup_base_path = os.path.join(
                os.getenv("HOME"), "datasets", "bad_data_backup"
            )
        else:
            backup_base_path = os.path.abspath(backup_base_path)

        assert os.path.exists(backup_base_path), (
            f"Resolved `backup_base_path` as {backup_base_path}, but it does not exist."
        )

        assert relative_cutoff is not None, (
            "Provided source and base backup paths for bad data buckup but `relative_cutoff` is None."
        )

        backup_bad_data = True

    try:
        dataset._dry_iter = True
        for sample in dataset:
            pass
    except (ValueError, ArrowInvalid):
        if not backup_bad_data:
            err_msg = (
                "\nWARNING: Got broken dataset that couldn't iterate through, "
                "but no source path was provided, so not backuping it. "
                "For a backup during checking please provide source and backup paths.\n"
            )
            print(err_msg)
            exit()

        backup_relative_path = source_path.split("/")[relative_cutoff:]
        backup_target_path = os.path.join(
            backup_base_path, *backup_relative_path
        )

        print(
            f"\nWARNING: Moving broken data from {source_path} to {backup_target_path}\n"
        )

        os.makedirs(str(Path(backup_target_path).parent), exist_ok=True)
        shutil.move(source_path, backup_target_path)

        exit()

    finally:
        dataset._dry_iter = False
