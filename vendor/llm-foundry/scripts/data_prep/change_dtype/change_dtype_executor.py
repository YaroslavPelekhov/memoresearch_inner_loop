import json
import os
import shutil
import subprocess as sp
from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import ThreadPool

from pathlib import Path
from typing import List, Dict


def parse_args():
    """
    Process CLI arguments.
    """
    parser = ArgumentParser(
        description="Process dataset part with provided using separate executor."
    )
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help="Path to a script to use for tokenization",
    )
    parser.add_argument(
        "-e",
        "--executor",
        type=int,
        required=True,
        help="Executor id for the job. Each executor precesses each own path.",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to a file with prepared task splits for dataset tokenization.",
    )
    parser.add_argument(
        "--script-additional-params",
        type=str,
        default="",
        help='A comma separated list of additional script arguments: "ARG1=VAL1,ARG2=VAL2,...". If not provided, defaults to an empty string.',
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair broken shards (without index.json) and continue with empty dirs",
    )

    parser.add_argument(
        "-n",
        "--n-proc",
        type=int,
        required=True,
        help="Number of processes to use for tokenization.",
    )

    return parser.parse_args()


def process_dataset(
    process_path_tuple: tuple, script: str, script_kwargs: Dict[str, str]
):
    """
    Launch processing script for target path pair as a subprocess.
    """
    source_path = process_path_tuple[0]
    out_path = process_path_tuple[1]
    try:
        processing_command_list = [
            "python",
            script,
            "--path",
            source_path,
            "--out_root",
            out_path,
        ]
        for key, val in script_kwargs.items():
            processing_command_list.extend([f"--{key}", str(val)])
        print(" ".join(processing_command_list))
        out = sp.run(processing_command_list, check=True, text=True)

        # print last 3000 characters
        print(processing_command_list, f'\n{"---"*3}\n', out[-3000:], f'\n{"---"*3}\n')
    except Exception as e:
        print("Tuple:", process_path_tuple)
        print("Exception:", e)
        raise

    return 1


def main(args):
    """
    Process data split for a provided executor id. Launches a number of subprocesses
    with provided processing script (`script`) and tokenization options.

    Parameters
    ----------
    args:
        Parsed CLI arguments using `argparse`. Expected to have following arguments:

        - `script` -- path to a script used to process dataset,
        - `executor_id` -- ID of an executor used to select the chunk to process,
        - `task_file` -- path to a task file with executors processing chunks,
        - `n_proc` -- number of processes to use for processing.
    """
    script = args.script
    executor_id = args.executor
    task_file = args.task_file
    script_additional_params = args.script_additional_params

    n_proc = args.n_proc
    print(args.repair)

    assert os.path.exists(script), f"Provided script does not exist: {script}!"
    assert os.path.exists(
        task_file
    ), f"Provided tasks file does not exist: {task_file}!"

    with open(task_file, "r") as file:
        tasks_data = json.load(file)

    script_kwargs = {
    }
    if script_additional_params != "":
        script_additional_params = script_additional_params.split(",")
        script_additional_params = [
            param.split("=") for param in script_additional_params
        ]
        for param_name, param_val in script_additional_params:
            script_kwargs[param_name] = param_val

    process_dataset_updated = partial(
        process_dataset, script=script, script_kwargs=script_kwargs
    )

    executor_tasks: List[tuple] = tasks_data[executor_id]
    if args.repair:
        print("repair tasks")
        executor_tasks = [
            task
            for task in executor_tasks
            if (
                not Path(task[1]).exists()
                or (
                    Path(task[1]).is_dir()
                    and not (Path(task[1]) / "index.json").exists()
                )
            )
        ]
        for task in executor_tasks:
            shutil.rmtree(task[1], ignore_errors=True)

    print("executor_tasks:", executor_tasks)
    with ThreadPool(n_proc) as pool:
        pool.map(process_dataset_updated, executor_tasks, 1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
