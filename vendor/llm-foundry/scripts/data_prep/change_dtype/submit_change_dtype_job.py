import json
import os
from argparse import ArgumentParser
from pathlib import Path

import client_lib


def parse_args():
    """
    Process CLI arguments.
    """
    parser = ArgumentParser(
        description="Prepare data in MosaicML StreamingDataset format using provided tokenizer."
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=False,
        default=None,
        help="Path to the original data for processing.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=False,
        default=None,
        help="Output path where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=False,
        default=None,
        help="Name of the task. Will be used to save JSON file with the task: files to process for each executor.",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        required=False,
        default=None,
        help="Path to JSON file with the task: files to process for each executor.",
    )
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help="Path to a process script that is used to tokenize data.",
    )
    parser.add_argument(
        "--script-additional-params",
        type=str,
        default="",
        help='A comma separated list of additional script arguments: "ARG1=VAL1,ARG2=VAL2,...". If not provided, defaults to an empty string.',
    )
    parser.add_argument(
        "-e",
        "--num-executors",
        type=int,
        default=1,
        help="Number of executors that will process dataset. Each executor processes their own part. Default: 1.",
    )
    parser.add_argument(
        "-n",
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in for each executor. Determines number of processes used by each executor to tokenize data: n_proc = 16 * n_gpus. Defaults to 1.",
    )
    parser.add_argument(
        "--repair",
        action='store_true',
        help='Repair broken shards (without index.json) and continue with empty dirs',
    )
    parser.add_argument(
        "--base-image",
        required=True,
        help='Base image with pre-installed requirements',
    )

    return parser.parse_args()


def prepare_data_splits(
    input_path: str, output_path: str, task_name: str, num_executors: int
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
    num_executors: int
        Number of executors to use for data processing. Each executor processes each own data split.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Provided input path does not exist: {input_path}!")

    assert os.path.isdir(input_path), f'Input path must be dir, provided {input_path}!'
    source_paths = set()
    for root, dirs, files in os.walk(input_path):
        for file in files:
            source_paths.add(root)
    source_paths = sorted(list(source_paths))
    dest_paths = [path.replace(input_path, output_path) for path in source_paths]

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


def submit_process_jobs(
    script: str,
    task_file: str,
    script_additional_params: str,
    num_executors: int,
    n_gpus: int,
    repair: bool,
    base_image: str,
):
    """
    Submit data processing jobs for each executor.

    Parameters
    ----------
    script: str
        Path to a script to use for data processing.
    task_file: str
        Path to a task file with input and output processing processing path pairs splits for executors.
    num_executors: int
        Number of executors to use for data processing. Determines the number of jobs submitted.
    n_gpus: int
        Number of GPUs for each job node. Cannot be more than 8 and determines the number of processes
        as n_proc = 16 * n_gpus.
    """
    assert os.path.exists(
        script
    ), f"Provided processing script does not exist: {script}!"

    n_gpus = min(8, n_gpus)
    N_WORKERS = 1
    n_cpus = 16 * n_gpus

    CODE_PATH = str(Path(__file__).absolute().parent.parent.parent.parent)
    external_code_libraries = [
        # repo root
        CODE_PATH,
        # submodules
        f"{CODE_PATH}/contrib/composer",
        f"{CODE_PATH}/contrib/streaming",
    ]
    assert all(os.path.exists(external_code_library) for external_code_library in external_code_libraries)

    script = os.path.abspath(script)

    launch_script_path = str(Path(__file__).absolute().parent / "change_dtype_executor.py")
    for executor_id in range(0, num_executors):
        job_script_params = {
            "script": script,
            "executor": executor_id,
            "task-file": task_file,
            "n-proc": n_cpus,
        }
        if script_additional_params != "":
            job_script_params["script-additional-params"] = script_additional_params
        job_script = ["python", launch_script_path]
        for param_name, param_val in job_script_params.items():
            job_script.extend([f"--{param_name}", str(param_val)])
        if repair:
            job_script.extend(["--repair"])
        job_script = " ".join(job_script)

        job_desc = f"Data tokenization job {executor_id+1}/{num_executors} | task file: {task_file}"

        print("Submitting job with the following script:")
        print(job_script)
        job = client_lib.Job(
            script=job_script,
            base_image=base_image,
            job_desc=job_desc,
            n_workers=N_WORKERS,
            processes_per_worker=1,
            instance_type=f"a100.{n_gpus}gpu",
            region="SR004",
            type="binary",
            preflight_check=False,
            pytorch_use_env=True,
            env_variables={
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                "PYTHONPATH": ":".join(external_code_libraries),
            },
        )
        print(f"[{executor_id+1}/{num_executors}] job submitted:", job.submit())


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    script = args.script
    task_name = args.task_name
    num_executors = args.num_executors
    n_gpus = args.n_gpus
    script_additional_params = args.script_additional_params

    print("Preparing executors splits...")
    task_file_path = args.task_file
    if not task_file_path:
        assert input_path
        assert output_path
        assert task_name
        task_file_path = prepare_data_splits(
            input_path, output_path, task_name, num_executors
        )

    print("Submitting processing jobs...")
    submit_process_jobs(
        script=script,
        task_file=task_file_path,
        num_executors=num_executors,
        n_gpus=n_gpus,
        script_additional_params=script_additional_params,
        repair=args.repair,
        base_image=args.base_image
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
