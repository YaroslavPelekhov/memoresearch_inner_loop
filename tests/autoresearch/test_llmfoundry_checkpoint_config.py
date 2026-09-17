"""Regression coverage for checkpoint-safe LLM Foundry config objects."""

import sys
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf
import pytest

from autoresearch.lmfoundry_compat import install as install_lmfoundry_compat

torch = pytest.importorskip("torch")
try:
    install_lmfoundry_compat()
except ModuleNotFoundError as exc:
    if exc.name == "composer" or (exc.name or "").startswith("composer."):
        pytest.skip("Composer is not installed", allow_module_level=True)
    raise

VENDORED_LLM_FOUNDRY = Path(__file__).parents[2] / "vendor" / "llm-foundry"
sys.path.insert(0, str(VENDORED_LLM_FOUNDRY))

from llmfoundry.utils.builders import build_optimizer, build_scheduler  # noqa: E402


def _contains_omegaconf(value: object) -> bool:
    if isinstance(value, (DictConfig, ListConfig)):
        return True
    if isinstance(value, dict):
        return any(_contains_omegaconf(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_omegaconf(item) for item in value)
    return False


def test_optimizer_and_scheduler_do_not_retain_omegaconf_containers() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(
        model,
        "decoupled_adamw",
        OmegaConf.create(
            {
                "lr": 0.001,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 1.0e-4,
            }
        ),
    )
    scheduler = build_scheduler(
        OmegaConf.create(
            {
                "name": "stacked",
                "schedule_rows": [
                    [1024, 1.0, "linear"],
                    [4096, 0.1, "cosine"],
                ],
            }
        )
    )

    assert not _contains_omegaconf(optimizer.param_groups)
    assert not _contains_omegaconf(vars(scheduler))
