import logging
from pathlib import Path
from typing import Optional

from composer.callbacks.checkpoint_saver import get_current_checkpoint_save
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist, TimerType, callback_timer
from omegaconf import DictConfig, OmegaConf
from contrib.job_submitter.job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF

local_logger = logging.getLogger(__name__)


PAM_KEYS = {
    "run_name",
    "exp_name",
    "user",
    "tensorboard_path",
    "max_seq_len",
    "autometrics",
}


class Autometrics(Callback):
    def __init__(self, callback_config: DictConfig):
        self.callback_config = callback_config
        self.eval_interval = Time.from_input(
            self.callback_config["eval_interval"], TimeUnit.BATCH
        )
        self.stable_save_interval = self.callback_config.get("stable_save_interval")
        if self.stable_save_interval is not None:
            self.check_interval = Time.from_input(
                self.stable_save_interval, TimeUnit.BATCH
            )
        else:
            self.check_interval = self.eval_interval

        self.user: Optional[str] = self.callback_config.get("user")
        self.nnodes: int = int(self.callback_config.get("nnodes"))
        self.sglang_tag = self.callback_config.get("sglang_tag", "opensource")
        self.saved_checkpoint_path: Optional[str] = None

    def _get_checkpoint_directory_name(
        self, checkpoint_save: Optional[dict]
    ) -> Optional[str]:
        if not checkpoint_save:
            return None

        saved_checkpoint_path = checkpoint_save.get("saved_path")
        saved_checkpoint_batch_value = checkpoint_save.get(
            "saved_checkpoint_batch_value"
        )

        if saved_checkpoint_path is None or saved_checkpoint_batch_value is None:
            return None

        if saved_checkpoint_path == self.saved_checkpoint_path:
            return None

        interval_value = int(self.check_interval.value)
        batch_value = int(saved_checkpoint_batch_value)

        if batch_value % interval_value == 0:
            return saved_checkpoint_path

        return None

    def _save_pam_config_to_artifacts(self, path_to_checkpoint_dir: Path) -> None:
        artifacts_path = Path(path_to_checkpoint_dir) / "artifacts"
        if not artifacts_path.exists():
            artifacts_path.mkdir(parents=True, exist_ok=True)
        pam_config_path = artifacts_path / "pam_config.yaml"
        llm_foundry_config_path = artifacts_path / "llm_foundry_config.yaml"
        with open(llm_foundry_config_path, "r") as f:
            llm_foundry_config = OmegaConf.load(f)
        pam_config = OmegaConf.create({})
        for key in PAM_KEYS:
            pam_config[key] = llm_foundry_config.get(key)
        OmegaConf.save(config=pam_config, f=pam_config_path)

    def _handle_autometrics_on_checkpoint(self, state: State) -> None:
        saved_checkpoint_path = self._get_checkpoint_directory_name(
            get_current_checkpoint_save(state)
        )
        if not saved_checkpoint_path:
            return

        path_to_checkpoint_dir = Path(saved_checkpoint_path).parent.parent

        if dist.get_global_rank() == 0:
            self._save_pam_config_to_artifacts(path_to_checkpoint_dir)
            AutometricsJobLLMF(
                checkpoint_dir=path_to_checkpoint_dir,
                user_login=self.user,
                exp_name=f"Autometrics run for checkpoint {path_to_checkpoint_dir}",
                n_workers=self.nnodes,
                max_retry=3,
                slurm_submit_type="api",
                sglang_tag=self.sglang_tag,
                debug=self.callback_config.get("debug", False),
            ).submit()
            self.saved_checkpoint_path = saved_checkpoint_path

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def batch_end(self, state: State, logger: Logger):
        try:
            if dist.get_global_rank() != 0:
                return
            local_logger.info("Submitting autometrics job on after_checkpoint_save")
            self._handle_autometrics_on_checkpoint(state)
        except Exception as e:
            local_logger.error(
                f"Error in autometrics callback (after_checkpoint_save): {e}"
            )
