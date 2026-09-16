import json
import os
import subprocess as sp
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    """
    Process CLI arguments.
    """
    parser = ArgumentParser(
        description="Prepare data in MosaicML StreamingDataset format using provided tokenizer."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to a config YAML file with all the necessary fields."
    )

    return parser.parse_args()


def prepare_data_splits(
    input_path: str,
    output_path: str,
    task_name: str,
    num_executors: int = 1,
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

    Returns
    -------
    str:
        Path to saved task file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Provided input path does not exist: {input_path}!")

    assert os.path.isdir(input_path), f"Input path must be dir, provided {input_path}!"
    source_paths = set()
    for root, dirs, files in os.walk(input_path):
        if len(dirs) > 0 and len(files) > 0:
            error_msg = (
                f"There are some files and dirs in the root directory {root}, but "
                "it supposed to have either only files or only dirs. Please, check it up."
            )
            raise RuntimeError(error_msg)

        if len(files) > 0:
            for file in files:
                source_paths.add(os.path.join(root, file))

    source_paths = tuple(sorted(source_paths))
    dest_paths = tuple(
        path.replace(input_path, output_path).replace(".tar.gz", "")
        for path in source_paths
    )

    task_data = []
    for i in range(len(source_paths)):
        task_data.append((source_paths[i], dest_paths[i]))

    exucutors_partitions = []
    partition_size = len(task_data) // num_executors + 1
    for i in range(num_executors):
        partition = task_data[i * partition_size : (i + 1) * partition_size]
        exucutors_partitions.append(partition)

    tasks_folder = Path(__file__).parent / "untar_tasks"
    if not tasks_folder.exists():
        os.makedirs(tasks_folder, exist_ok=True)

    task_file_path = str(tasks_folder / f"{task_name}.json")
    with open(task_file_path, "w") as fout:
        json.dump(exucutors_partitions, fout)

    return task_file_path


def submit_process_jobs(
    source_dir: str,
    dest_dir: str,
    task_file_path: str,
    n_proc: int = 1,
    num_executors: int = 1,
):
    root_dir = str(Path(__file__).absolute().parent.parent.parent)
    export_envs = (
        f"ROOT_DIR={root_dir}",
        f"SOURCE_DIR={source_dir}",
        f"DEST_DIR={dest_dir}",
        f"TASK_FILE_PATH={task_file_path}",
        f"N_PROC={n_proc}",
    )
    export_str = ",".join(export_envs)

    sbatch_command = [
        "sbatch",
        "--export",
        export_str,
        f"--nodes={num_executors}",
        f"--gpus-per-node=8",
        "slurm-sbatch-untar.sh"
    ]
    try:
        out = sp.run(sbatch_command, check=True, text=True)
    except Exception as e:
        print("Exception:", e)
        raise


def main(args):
    config_path = args.config
    with open(config_path) as fin:
        config = json.load(fin)

    source_dir = config["source_dir"]
    dest_dir = config["dest_dir"]
    task_name = config["task_name"]
    num_executors = config["num_executors"]
    n_proc = config["n_proc"] # 2 CPU cores per GPU

    task_file_path = prepare_data_splits(
        source_dir,
        dest_dir,
        task_name=task_name,
        num_executors=num_executors
    )

    print(f"Submitting untar jobs for {source_dir} directory...")
    submit_process_jobs(
        source_dir=source_dir,
        dest_dir=dest_dir,
        task_file_path=task_file_path,
        n_proc=n_proc,
        num_executors=num_executors,
    )


if __name__ == "__main__":
    main(parse_args())
