from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import requests
import wandb
from composer.callbacks.checkpoint_saver import get_current_checkpoint_save
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional
import re
import subprocess as sp
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
PIPE_TIMEOUT = 60 * 15

JOBS_DICT = {
    "bbh": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "bbh",
        "num_fewshot": "3",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "arc": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "arc_easy,arc_challenge",
        "num_fewshot": "25",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "truthfulqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "drop": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "drop",
        "num_fewshot": "3",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "race": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "race",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "triviaqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "triviaqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "agieval": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "agieval",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "commonsense_qa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "commonsense_qa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "lambada": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "lambada",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "openbookqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "openbookqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "piqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "piqa",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "gsm8k": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "gsm8k",
        "num_fewshot": "4",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mathqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mathqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mgsm": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mgsm_direct,mgsm_cot_native",
        "num_fewshot": "8",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "minerva_math": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "minerva_math",
        "num_fewshot": "4",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "gpqa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "gpqa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "humaneval": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "humaneval",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "mbpp": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "mbpp",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xcopa": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xcopa",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xwinograd": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xwinograd",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "xstorycloze": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "xstorycloze",
        "num_fewshot": "0",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "pawsx": {
        "model_type": "vllm",
        "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=8,gpu_memory_utilization=0.5",
        "tasks": "pawsx",
        "num_fewshot": "5",
        "batch_size": "auto",
        "n_gpus": "8",
    },
    "longbench": {
        "max_length": "8192",
        "model_dtype": "fp16",
        "n_gpus": "8",
        "model_name": "gigachat-gigar-vllm",
    },
    "passkey": {"model_dtype": "bf16", "n_gpus": "8"},
    "sam": {
        "tasks": "sam",
        "pass_obm": False,  # Пропустить замер метрик OBM
        "pass_obb": False,  # Пропустить этап замера обб
        "pass_judge": False,  # Пропустить сбор бакетов Judge-метрик
        "obb_datasets": [],  # Метрики обб, если не указывать это поле, то будут замеряться дефолтные
        "obm_datasets": [],  # Метрики обб, если не указывать это поле, то будут замеряться дефолтные
        "obm_groups": [],
    },
}


class JobCallbackDispatcher(Callback):
    def __init__(self, all_kwargs: DictConfig):
        self.jdc_config = all_kwargs
        self.export_path = self._resolve_export_path()
        self._validate_export_path()
        self.eval_interval = Time.from_input(
            self.jdc_config.pop("eval_interval"), TimeUnit.BATCH
        )
        self.last_checkpoint = None

    def _resolve_export_path(self) -> str:
        """Resolves the export path from config or environment variable."""
        if "signal_files_path" in self.jdc_config:
            export_path = self.jdc_config.pop("signal_files_path")
        elif "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES" in os.environ:
            export_path = os.environ["JOB_DISPATCHER_CALLBACK_SIGNAL_FILES"]
        else:
            raise ValueError("Can't find JOB_DISPATCHER_CALLBACK path!")
        print("EXPORTPATH_HERE", export_path)
        return export_path

    def _validate_export_path(self):
        """Validates that the export path exists."""
        if not Path(self.export_path).exists():
            raise FileNotFoundError(f"{self.export_path} path does not exist!")

    def after_checkpoint_save(self, state: State, logger: Logger):
        self._handle_checkpoint_update(state)

    def _handle_checkpoint_update(self, state: State):
        """Handles checkpoint updates and triggers export if necessary."""
        checkpoint_save = get_current_checkpoint_save(state)
        if not checkpoint_save:
            return

        saved_checkpoint = checkpoint_save["saved_path"]
        saved_checkpoint_batch_value = checkpoint_save["saved_checkpoint_batch_value"]

        if (
            saved_checkpoint != self.last_checkpoint
            and saved_checkpoint_batch_value % self.eval_interval.value == 0
        ):
            self.last_checkpoint = saved_checkpoint
            self.jdc_config.ckpt_path = str(Path(self.last_checkpoint).parent)
            if dist.get_global_rank() == 0:
                self.export_json()

    def export_json(self):
        filename = tempfile.mktemp(suffix=".json", dir=self.export_path)
        resulting_job_dict = dict()
        jobs_dict_config = om.to_container(self.jdc_config, resolve=True)
        print(jobs_dict_config, file=sys.stderr)
        if jobs_dict_config.get("job_params_override", None) is not None:
            resulting_job_dict.update(jobs_dict_config["job_params_override"])

        resulting_job_dict.update(jobs_dict_config["job_params_override"])
        customized_benchmarcks = set()
        for key, val in resulting_job_dict.items():
            if key in ["longbench", "passkey"]:
                customized_benchmarcks.add(key)
            else:
                if "tasks" in val:
                    customized_benchmarcks.update(val["tasks"].split(","))
        if jobs_dict_config.get("job_list", None) is not None:
            for job_group in jobs_dict_config["job_list"].values():
                for job in job_group.split():
                    if job not in JOBS_DICT:
                        raise ValueError(
                            f"Metric '{job}' has no default configuration and should not be in 'job_list' argument. Please provide custom configuration for '{job}' in 'job_params_override'."
                        )
                    if job not in customized_benchmarcks:
                        if job in ["longbench", "passkey"]:
                            resulting_job_dict[job] = JOBS_DICT[job]
                        else:
                            default = deepcopy(JOBS_DICT[job])
                            tasks = ",".join(
                                [
                                    task
                                    for task in default["tasks"].split(",")
                                    if task not in customized_benchmarcks
                                ]
                            )
                            if tasks:
                                default["tasks"] = tasks
                                if job != "sam":
                                    resulting_job_dict[
                                        "lm_eval_" + job + "_default_config"
                                    ] = default
                                else:
                                    resulting_job_dict[job] = default

        jobs_dict_config["jobs"] = resulting_job_dict
        if "job_list" in jobs_dict_config:
            del jobs_dict_config["job_list"]
        if "job_params_override" in jobs_dict_config:
            del jobs_dict_config["job_params_override"]

        jobs_dict_config["save_hf_in_ckpt_dir"] = self.jdc_config.get(
            "save_hf_in_ckpt_dir", False
        )
        num_attention_heads = jobs_dict_config["num_attention_heads"]
        if "longbench" in jobs_dict_config["jobs"]:
            for n_gpus in range(
                min(8, int(jobs_dict_config["jobs"]["longbench"]["n_gpus"])), 0, -1
            ):
                if num_attention_heads % n_gpus == 0:
                    jobs_dict_config["jobs"]["longbench"]["n_gpus"] = str(n_gpus)
                    break
        if "passkey" in jobs_dict_config["jobs"]:
            for n_gpus in range(
                min(8, int(jobs_dict_config["jobs"]["passkey"]["n_gpus"])), 0, -1
            ):
                if num_attention_heads % n_gpus == 0:
                    jobs_dict_config["jobs"]["passkey"]["n_gpus"] = str(n_gpus)
                    break
        for key, val in jobs_dict_config["jobs"].items():
            if (
                key not in ["longbench", "passkey"]
                and "n_gpus" in val
                and "model_args" in val
            ):
                for n_gpus in range(min(8, int(val["n_gpus"])), 0, -1):
                    if num_attention_heads % n_gpus == 0:
                        params = val["model_args"].split(",")
                        for i, param in enumerate(params):
                            if param.startswith("tensor_parallel_size="):
                                params[i] = "tensor_parallel_size=" + str(n_gpus)
                                break
                        else:
                            params.append("tensor_parallel_size=" + str(n_gpus))
                        break
                val["model_args"] = ",".join(params)

        # hash_digest = hashlib.md5(json.dumps(jobs_dict_config, sort_keys=True).encode('utf-8')).hexdigest()
        # filename = Path(self.export_path) / f"{hash_digest}.json"

        logger.info(f"Start writing callback .json config in {filename}")
        with Path(filename).open("w", encoding="utf-8") as f:
            json.dump(jobs_dict_config, f, ensure_ascii=False, indent=4)


class JobCallbackDispatcherWEB(Callback):
    def __init__(self, all_kwargs: DictConfig):
        self.jdcw_config = all_kwargs
        self.eval_interval = Time.from_input(
            self.jdcw_config.pop("eval_interval"), TimeUnit.BATCH
        )
        self.last_checkpoint = None

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job_dispatcher_web")
        self._send_model_future = None

    def after_checkpoint_save(self, state: State, logger: Logger):
        if dist.get_global_rank() != 0:
            return
        self._handle_checkpoint_update(state)

    def _handle_checkpoint_update(self, state: State):
        """Handles checkpoint updates and triggers export if necessary."""
        checkpoint_save = get_current_checkpoint_save(state)
        if not checkpoint_save:
            return

        saved_checkpoint = checkpoint_save["saved_path"]
        saved_checkpoint_batch_value = checkpoint_save["saved_checkpoint_batch_value"]

        if (
            saved_checkpoint != self.last_checkpoint
            and saved_checkpoint_batch_value % self.eval_interval.value == 0
        ):
            self.last_checkpoint = saved_checkpoint
            checkpoint_dir = str(Path(self.last_checkpoint).parent)
            try:
                self.export_json(checkpoint_dir)
            except Exception as e:
                logger.error(f"Failed to export json in job dispatcher callback for path {checkpoint_dir}: {e}")

            self._submit_send_model(checkpoint_dir)

    def _replace_checkpoint_placeholders(
        self, config_dict: dict, checkpoint_path: str, wandb_run=None
    ) -> None:
        checkpoint_dir = str(Path(checkpoint_path).parent)

        placeholders = {
            "{CHECKPOINT_PATH}": checkpoint_path,
            "{CHECKPOINT_DIR}": checkpoint_dir,
        }

        if wandb_run is not None:
            placeholders["{WANDB_ENTITY}"] = wandb_run.entity
            placeholders["{WANDB_PROJECT}"] = wandb_run.project
            placeholders["{WANDB_RUN_ID}"] = wandb_run.id

        def replace(obj):
            if isinstance(obj, str):
                for placeholder, replacement in placeholders.items():
                    obj = obj.replace(placeholder, replacement)
                return obj
            elif isinstance(obj, dict):
                return {k: replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace(v) for v in obj]
            return obj

        for key, value in config_dict.items():
            config_dict[key] = replace(value)

    def export_json(self, checkpoint_path: str):
        jobs_config = om.to_container(self.jdcw_config, resolve=True)
        jobs_config["checkpoint_path"] = checkpoint_path

        run = wandb.run  # текущий run в wandb

        self._replace_checkpoint_placeholders(
            jobs_config, checkpoint_path, wandb_run=run
        )

        if run is not None:
            jobs_config["wandb_entity"] = run.entity
            jobs_config["wandb_project"] = run.project
            jobs_config["wandb_run_id"] = run.id
        else:
            logger.info(
                "WandB is not initialized, skipping wandb-related configuration"
            )
        jobs_config["llmf_path"] = str(Path(__file__).parent.parent.parent)
        path = Path(checkpoint_path).parent / "artifacts/job-dispatcher"
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"CREATED PATH FOR ARTIFACTS {path}")

        job_dispatcher_url = os.environ.get("JOB_DISPATCHER_BASE_URL")

        endpoint_path = (
            "/jobs/run-custom-jobs"
            if "custom_pipeline" in jobs_config
            else "/jobs/run-jobs"
        )
        response = requests.post(
            url=f"{job_dispatcher_url}{endpoint_path}", json=jobs_config, verify=False
        )
        if response.status_code != 200:
            raise ValueError(
                f"job_dispatcher_callback_error {response.status_code}, {response.text}"
            )
        else:
            logger.info("Request sent to eval metrics via job_dispatcher")

    @staticmethod
    def _parse_checkpoint_path(hf_checkpoint_path: Path) -> tuple[str, str, int, int]:
        for candidate in (hf_checkpoint_path, *hf_checkpoint_path.parents):
            name = candidate.name

            match = re.fullmatch(r"ep(\d+)_ba(\d+)", name)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                model_dir = candidate.parent
                return str(model_dir), model_dir.name, epoch, step

            match = re.fullmatch(r"global_step_(\d+)", name)
            if match:
                epoch = 0
                step = int(match.group(1))
                model_dir = candidate.parent
                return str(model_dir), model_dir.name, epoch, step

        return str(hf_checkpoint_path), hf_checkpoint_path.name, 0, 0

    @staticmethod
    def _get_next_s3_index(
        s3_dest: str,
        path_base: str,
        file_type: str,
        rclone_path: str,
        rclone_config: str,
    ) -> Optional[int]:
        proc = sp.run(
            [rclone_path, "--config", rclone_config, "lsf", s3_dest],
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )

        if file_type == "code":
            pattern = re.compile(rf"^{re.escape(path_base)}_(\d{{4}})\.tar$")
        elif file_type == "config":
            pattern = re.compile(rf"^{re.escape(path_base)}_(\d{{4}})\.yaml$")
        else:
            logger.error(f"Failed to getting existing llm-foundry code versions from s3: Unsupported s3 file type")
            return None

        max_index = 0
        for line in proc.stdout.splitlines():
            m = pattern.match(line.strip())
            if m:
                max_index = max(max_index, int(m.group(1)))
        return max_index + 1

    @staticmethod
    def _is_job_dispatcher_available(job_dispatcher_url: Optional[str]) -> bool:
        if not job_dispatcher_url:
            logger.warning("JOB_DISPATCHER_BASE_URL is not set; skipping job-dispatcher call")
            return False
        try:
            requests.get(job_dispatcher_url, verify=False, timeout=15)
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"job-dispatcher is not reachable at {job_dispatcher_url}: {e}; skipping")
            return False

    @staticmethod
    def _is_s3_available(
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ) -> bool:
        if not s3_dest or not rclone_path or not rclone_config:
            logger.warning("s3 destination/rclone settings are not configured; skipping s3 access")
            return False
        try:
            sp.run(
                [rclone_path, "--config", rclone_config, "lsf", "--max-depth", "1", s3_dest],
                capture_output=True,
                text=True,
                check=True,
                timeout=DEFAULT_TIMEOUT,
            )
            return True
        except Exception as e:
            logger.warning(f"s3 is not reachable at {s3_dest}: {e}; skipping s3 access")
            return False

    @staticmethod
    def _delete_s3_file(
        s3_file_path: str,
        rclone_path: str,
        rclone_config: str,
    ) -> None:
        cleanup_cmd = [
            rclone_path,
            "--config", rclone_config,
            "deletefile",
            "--s3-no-check-bucket",
            s3_file_path,
        ]
        sp.run(
            cleanup_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )

    @staticmethod
    def _upload_file_to_s3(
        filename: str,
        path: str,
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ):
        s3_file_path = f"{s3_dest}/{filename}"
        logger.info(f"Uploading {path} to {s3_file_path}")
        script = [
            rclone_path,
            "--config", rclone_config,
            "-P", "copyto",
            "--s3-no-check-bucket",
            "--transfers", "16",
            "--buffer-size", "256Mi",
            "--multi-thread-streams", "16",
            "--s3-disable-checksum",
            "--ignore-checksum",
            "--fast-list",
            "--no-update-modtime",
            str(path),
            s3_file_path,
        ]
        try:
            sp.run(
                script,
                capture_output=True,
                text=True,
                check=True,
                timeout=PIPE_TIMEOUT,
            )
        except Exception:
            JobCallbackDispatcherWEB._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise

        logger.info(f"Successfully uploaded {path} to {s3_file_path}")
        return s3_file_path

    @staticmethod
    def _tar_and_upload_llm_foundry_code(
        checkpoint_dir_escaped: str,
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ) -> Optional[str]:
        archive_base = f"llm_foundry_{checkpoint_dir_escaped}"

        index = JobCallbackDispatcherWEB._get_next_s3_index(s3_dest, archive_base, "code", rclone_path, rclone_config)
        if index is None:
            return None

        archive_name = f"{archive_base}_{index:04d}.tar"
        s3_file_path = f"{s3_dest}/{archive_name}"

        llm_foundry_path = Path(__file__).resolve().parents[2]
        if not llm_foundry_path.exists():
            logger.warning(f"Could not infer llm-foundry path (tried: {llm_foundry_path})")
            return None

        logger.info(f"Streaming tar of {llm_foundry_path} -> {s3_file_path}")

        tar_cmd = [
            "tar",
            "-cf", "-",
            "-C", str(llm_foundry_path.parent),
            llm_foundry_path.name,
        ]
        rclone_cmd = [
            rclone_path,
            "--config", rclone_config,
            "rcat",
            "--s3-no-check-bucket",
            "--s3-disable-checksum",
            "--ignore-checksum",
            s3_file_path,
        ]

        tar_proc = sp.Popen(tar_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        try:
            rclone_proc = sp.Popen(
                rclone_cmd,
                stdin=tar_proc.stdout,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
        except Exception:
            tar_proc.kill()
            tar_proc.wait()
            raise

        tar_proc.stdout.close()

        tar_err_chunks: list[bytes] = []
        def _drain_tar_stderr():
            try:
                assert tar_proc.stderr is not None
                for chunk in iter(lambda: tar_proc.stderr.read(4096), b""):
                    tar_err_chunks.append(chunk)
            except Exception:
                pass
        tar_stderr_drain = threading.Thread(
            target=_drain_tar_stderr,
            name="tar-stderr-drain",
            daemon=True,
        )
        tar_stderr_drain.start()

        try:
            _, rclone_err = rclone_proc.communicate(timeout=PIPE_TIMEOUT)
            tar_proc.wait(timeout=DEFAULT_TIMEOUT)
        except sp.TimeoutExpired:
            rclone_proc.kill()
            tar_proc.kill()
            rclone_proc.communicate()
            tar_proc.wait()
            tar_stderr_drain.join(timeout=5)
            JobCallbackDispatcherWEB._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise

        tar_stderr_drain.join(timeout=5)
        tar_err = b"".join(tar_err_chunks)

        if tar_proc.returncode != 0 or rclone_proc.returncode != 0:
            JobCallbackDispatcherWEB._delete_s3_file(s3_file_path, rclone_path, rclone_config)
            raise sp.CalledProcessError(
                rclone_proc.returncode if rclone_proc.returncode != 0 else tar_proc.returncode,
                rclone_cmd if rclone_proc.returncode != 0 else tar_cmd,
                stderr=tar_err.decode(errors='replace'),
            )

        logger.info(f"Successfully uploaded tar to {s3_file_path}")
        return s3_file_path

    @staticmethod
    def _upload_llm_foundry_config(
        hf_checkpoint_path: Path,
        checkpoint_dir_escaped: str,
        s3_dest: str,
        rclone_path: str,
        rclone_config: str,
    ) -> Optional[str]:
        config_base = f"llm_foundry_config_{checkpoint_dir_escaped}"

        index = JobCallbackDispatcherWEB._get_next_s3_index(s3_dest, config_base, "config", rclone_path, rclone_config)
        if index is None:
            return None

        config_s3_name = f"{config_base}_{index:04d}.yaml"
        config_nfs_path = f"{hf_checkpoint_path.parent}/artifacts/llm_foundry_config.yaml"

        s3_llmf_config_path = JobCallbackDispatcherWEB._upload_file_to_s3(config_s3_name, config_nfs_path, s3_dest, rclone_path, rclone_config)
        return s3_llmf_config_path

    @staticmethod
    def _make_checkpoint_dir_escaped(hf_checkpoint_path: Path) -> str:
        full_path = str(hf_checkpoint_path.parent)
        marker = "gigachat_checkpoints/"
        idx = full_path.find(marker)
        relative_path = full_path[idx + len(marker):] if idx >= 0 else full_path
        checkpoint_dir_escaped = relative_path.replace("/", "\\")
        return checkpoint_dir_escaped


    def send_model(self, checkpoint_path: str) -> None:
        config = om.to_container(self.jdcw_config, resolve=True)
        s3_dest = config.get("s3_dest")
        rclone_path = config.get("rclone_path")
        rclone_config = config.get("rclone_config")

        path = Path(checkpoint_path)
        model_dir, model_name, epoch, step = self._parse_checkpoint_path(path)

        checkpoint_dir_escaped = self._make_checkpoint_dir_escaped(path)

        if self._is_s3_available(s3_dest, rclone_path, rclone_config):
            llmf_code_s3_path = self._tar_and_upload_llm_foundry_code(checkpoint_dir_escaped, s3_dest, rclone_path, rclone_config)
            llmf_config_s3_path = self._upload_llm_foundry_config(path, checkpoint_dir_escaped, s3_dest, rclone_path, rclone_config)
        else:
            logger.warning("Skipping s3 uploads of llm-foundry code/config: s3 is not available in this environment")
            llmf_code_s3_path = None
            llmf_config_s3_path = None

        model_dict = {
            "nfs_path": model_dir,
            "name": model_name,
            "size": config.get("size"),
            "model_class": config.get("model_class"),
            "username": config.get("username"),
            "region": os.environ.get("MLS_JOB_REGION_NAME"),
            "step": step,
            "llmf_code_s3_path": llmf_code_s3_path,
            "llmf_config_s3_path": llmf_config_s3_path,
            "ml_space_job": {
                "job_name": os.environ.get("MLSPACE_JOB_NAME"),
                "workspace": os.environ.get("MLS_JOB_WORKSPACE_NAME"),
            }
        }

        run = wandb.run
        if run is not None:
            model_dict["wandb_entity"] = run.entity
            model_dict["wandb_project"] = run.project
            model_dict["wandb_run_id"] = run.id
        else:
            logger.info("WandB is not initialized, skipping wandb-related configuration")

        job_dispatcher_url = os.environ.get('JOB_DISPATCHER_BASE_URL')
        if not self._is_job_dispatcher_available(job_dispatcher_url):
            logger.warning("Skipping add_model_with_step: job-dispatcher is not available in this environment")
            return

        endpoint_path = "/ml_models/add_model_with_step"

        response = requests.post(
            url=f"{job_dispatcher_url}{endpoint_path}",
            json=model_dict,
            verify=False,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            logger.error(f"job_dispatcher_callback_error {response.status_code}, {response.text}")
            return

        logger.info("Request sent to add model via job_dispatcher")


    def _submit_send_model(self, checkpoint_dir: str) -> None:
        if self._send_model_future is not None and not self._send_model_future.done():
            logger.info("Previous send_model is still running; skipping new submission")
            return

        future = self._executor.submit(self.send_model, checkpoint_dir)
        def _log_exc(f):
            exc = f.exception()
            if exc is not None:
                logger.error(f"send_model failed: {exc}", exc_info=exc)
        future.add_done_callback(_log_exc)
        self._send_model_future = future
