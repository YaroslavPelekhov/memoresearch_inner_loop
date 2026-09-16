import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint as pp

import client_lib

from utils import prepare_data_splits, load_config, is_merged

CORES_PER_GPU = 8


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
        help="Path to a config JSON file with all the necessary fields."
    )

    return parser.parse_args()


def submit_process_jobs(
    script: str,
    tokenizer_path: str,
    task_file: str,
    num_executors: int,
    n_gpus: int,
    max_tokenized_len: int,
    script_additional_params: str,
    base_image: str,
):
    f"""
    Submit data processing jobs for each executor.

    Parameters
    ----------
    script: str
        Path to a script to use for data processing.
    tokenizer_path: str
        Path to a tokenizer folder in HF format.
    task_file: str
        Path to a task file with input and output processing processing path pairs splits for executors.
    num_executors: int
        Number of executors to use for data processing. Determines the number of jobs submitted.
    n_gpus: int
        Number of GPUs for each job node. Cannot be more than 8 and determines the number of processes
        as n_proc = {CORES_PER_GPU} * n_gpus.
    max_tokenized_len: int
        Maximux tokenization length to use.
    script_additional_params: str
        Additional params for tokenization script to specify options.
    base_image: str
        Base image that is used to launch tokenization job.
    """
    assert os.path.exists(script), f"Provided processing script does not exist: {script}!"
    assert os.path.exists(
        tokenizer_path
    ), f"Provided tokenizer path does not exist: {tokenizer_path}!"

    n_gpus = min(8, n_gpus)
    N_WORKERS = 1
    n_cpus = CORES_PER_GPU * n_gpus

    CODE_PATH = str(Path(__file__).absolute().parent.parent.parent)
    external_code_libraries = [
        # repo root
        CODE_PATH,
        # submodules
        f"{CODE_PATH}/contrib/composer",
        f"{CODE_PATH}/contrib/streaming",
    ]

    script = os.path.abspath(script)

    launch_script_path = str(Path(__file__).absolute().parent / "tokenize_executor.py")
    for executor_id in range(0, num_executors):
        job_script_params = {
            "script": script,
            "executor": executor_id,
            "task-file": task_file,
            "tokenizer": tokenizer_path,
            "max-tokenized-len": max_tokenized_len,
            "n-proc": n_cpus,
        }
        if script_additional_params != "":
            job_script_params["script-additional-params"] = script_additional_params
        job_script = ["python", launch_script_path]
        for param_name, param_val in job_script_params.items():
            job_script.extend([f"--{param_name}", str(param_val)])
        job_script = " ".join(job_script)

        job_desc = f"Data tokenization job {executor_id+1}/{num_executors} | task file: {task_file} | tokenizer: {tokenizer_path} | max tokenization len: {max_tokenized_len}"

        print("Submitting job with the following script:")
        print(job_script)
        job = client_lib.Job(
            script=job_script,
            base_image=base_image,
            job_desc=job_desc,
            n_workers=N_WORKERS,
            processes_per_worker=1,
            instance_type=f"a100.{n_gpus}gpu",
            priority_class="medium",
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
    config = load_config(args.config)

    print("Tokenization config:")
    pp(config)

    base_input_path = config["base_input_path"]
    base_output_path = config["base_output_path"]
    tokenizer_name = os.path.basename(config["tokenizer_path"])

    # TODO: Add per dataset specific params merge if needed
    script_additional_params = config["script_additional_params"]

    for dataset in config["datasets"]:
        dataset_config = config["datasets"][dataset]
        input_path = os.path.join(base_input_path, dataset)
        output_path = os.path.join(base_output_path, dataset)

        if is_merged(output_path):
            print(f"Dataset {dataset} is already processed and merged at {output_path}. Skipping it.")
            continue

        task_name = f"{dataset}_{tokenizer_name}_{config['max_tokenized_len']}"
        num_executors = dataset_config["num_executors"]
        task_file_path = prepare_data_splits(
            input_path,
            output_path,
            task_name,
            num_executors,
            dataset_config["per_file_task_split"]
        )

        script = os.path.abspath(dataset_config["script_path"])

        print(f"Submitting tokenization jobs for {dataset} dataset...")
        submit_process_jobs(
            script=script,
            tokenizer_path=config["tokenizer_path"],
            task_file=task_file_path,
            num_executors=num_executors,
            n_gpus=config["n_gpus"],
            max_tokenized_len=config["max_tokenized_len"],
            base_image=config["base_image"],
            script_additional_params=script_additional_params,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
