from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal, Union, Dict
from omegaconf import DictConfig, OmegaConf

class ConvertationConfig(BaseModel):
    class Config:
        extra = "forbid"

    base_image: Optional[str] = Field(default=None)
    tp_size: Optional[int] = Field(default=None)
    n_gpus: int = Field(8)
    mappers: Optional[List] = Field(default=None)

class MetricsLongbenchConfig(BaseModel):
    class Config:
        extra = "forbid"

    base_image: Optional[str] = Field(default=None)
    max_length: int = Field(..., ge=1)
    dtype: Literal["fp16", "fp32", "bf16"] = Field(..., alias="model_dtype")
    n_gpus: int = 1
    name: str = Field(..., alias="model_name")

class MetricsPasskeyConfig(BaseModel):
    class Config:
        extra = "forbid"

    base_image: Optional[str] = Field(default=None)
    device_map: Optional[str] = Field(default=None)
    dtype: Literal["fp16", "fp32", "bf16"] = Field(..., alias="model_dtype")
    n_gpus: int = 1

class MetricsLMEvalConfig(BaseModel):
    class Config:
        extra = "forbid"

    base_image: Optional[str] = Field(default=None)
    type: str = Field(..., alias="model_type")
    n_gpus: int = 1
    args: str = Field(..., alias="model_args")
    tasks: str
    num_fewshot: int
    batch_size: Union[str, int]

class MetricsBigCodeConfig(BaseModel):
    class Config:
        extra = "forbid"

    base_image: Optional[str] = Field(default=None)
    n_gpus: Literal[1] = Field(1)
    precision: Literal["fp16", "fp32", "bf16"]
    n_samples: int
    batch_size: int

class MetricsLMEvalGroupConfig(BaseModel):
    class Config:
        extra = "forbid"

    group_name: Literal["general", "math", "code", "multilingual"]

class Sam(BaseModel):
    class Config:
        extra = "forbid"
    tasks: str
    pass_obm: Optional[bool] = False
    pass_obb: Optional[bool] = True
    pass_judge: Optional[bool] = False
    obb_datasets: Optional[List] = []
    obm_datasets: Optional[List] = []
    obm_groups: Optional[List] = []


class JobCallbackDispatcherConfig(BaseModel):
    class Config:
        extra = "forbid"

    username: str
    signal_files_path: Optional[str] = Field(default=None)
    eval_interval: Union[int, str]
    tokenizer_path: str
    convertation: ConvertationConfig
    num_attention_heads: int
    save_hf_in_ckpt_dir: bool
    job_list: Optional[Dict[str, str]] = None
    job_params_override: Optional[Dict[str, Union[
        MetricsLongbenchConfig,
        MetricsPasskeyConfig,
        MetricsLMEvalConfig,
        MetricsBigCodeConfig,
        Sam
    ]]] = None

def validate_job_dispatcher_callback_config(cfg: DictConfig):
    if not hasattr(cfg, 'callbacks') or not hasattr(cfg.callbacks, 'job_dispatcher_callback'):
        return
    job_dispatcher_callback_dict = OmegaConf.to_container(
        cfg.callbacks.job_dispatcher_callback, resolve=True
    )
    try:
        JobCallbackDispatcherConfig.model_validate(job_dispatcher_callback_dict)
    except ValidationError as e:
        raise ValueError(f"Validation Error in Pretrain Autometrics Config: {e}")
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while validating Pretrain Autometrics Config: {e}"
        ) from e