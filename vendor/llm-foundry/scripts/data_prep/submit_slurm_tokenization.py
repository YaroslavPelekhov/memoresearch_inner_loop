import os
import subprocess as sp
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint as pp

from utils import prepare_data_splits, load_config, is_merged


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
    task_file_path: str,
    max_tokenized_len: int,
    num_executors: int = 1,
    n_proc: int = 1,
    script_additional_params: str = "",
):
    root_dir = str(Path(__file__).absolute().parent.parent.parent)
    executor_script_path = str(Path(__file__).absolute().parent / "tokenize_executor.py")
    export_envs = (
        f"ROOT_DIR={root_dir}",
        f"EXECUTOR_SCRIPT_PATH={executor_script_path}",
        f"TOKENIZATION_SCRIPT_PATH={script}",
        f"TOKENIZER_PATH={tokenizer_path}",
        f"TASK_FILE_PATH={task_file_path}",
        f"MAX_TOKENIZED_LEN={max_tokenized_len}",
        f"NUM_EXECUTORS={num_executors}",
        f"N_PROC={n_proc}",
        f"SCRIPT_ADDITIONAL_PARAMS={script_additional_params}",
    )
    export_str = ",".join(export_envs)

    sbatch_command = [
        "sbatch",
        "--export",
        export_str,
        f"--nodes={num_executors}",
        "slurm-sbatch-tokenization-start.sh"
    ]
    try:
        out = sp.run(sbatch_command, check=True, text=True)
    except Exception as e:
        print("Exception:", e)
        raise


def main(args):
    config_path = args.config
    config = load_config(config_path)

    print("Tokenization config:")
    pp(config)

    base_input_path = config["base_input_path"]
    base_output_path = config["base_output_path"]
    tokenizer_name = os.path.basename(config["tokenizer_path"])

    # TODO: Add per dataset specific params merge if needed
    script_additional_params = config["script_additional_params"]

    for dataset in config["datasets"]:
        input_path = os.path.join(base_input_path, dataset)
        output_path = os.path.join(base_output_path, dataset)

        if is_merged(output_path):
            print(f"Dataset {dataset} is already processed and merged at {output_path}. Skipping it.")
            continue

        task_name = f"{dataset}_{tokenizer_name}_{config['max_tokenized_len']}"
        num_executors = config["datasets"][dataset]["num_executors"]
        task_file_path = prepare_data_splits(
            input_path,
            output_path,
            task_name,
            num_executors,
            config["datasets"][dataset]["per_file_task_split"]
        )

        script = os.path.abspath(config["datasets"][dataset]["script_path"])

        print(f"Submitting tokenization jobs for {dataset} dataset...")
        submit_process_jobs(
            script=script,
            tokenizer_path=config["tokenizer_path"],
            task_file_path=task_file_path,
            num_executors=num_executors,
            n_proc=config["n_proc"],
            max_tokenized_len=config["max_tokenized_len"],
            script_additional_params=script_additional_params,
        )


if __name__ == "__main__":
    main(parse_args())
