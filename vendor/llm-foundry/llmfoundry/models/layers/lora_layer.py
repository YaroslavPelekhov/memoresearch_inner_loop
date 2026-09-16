import typing as tp
import logging
import math

import operator
import torch
import torch.nn as nn
from typing import Any, Callable, Literal, Optional, Union

logger = logging.getLogger(__name__)


from peft.tuners import LoraModel
from peft.tuners.lora import LoraLayer
from peft.utils.other import get_pattern_key
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_quantization_config,
)

from llmfoundry.models.layers.fc import ColumnParallelLinear, RowParallelLinear
from llmfoundry.models.layers.moe.experts import GroupedLlamaMLP, fused_swiglu
from llmfoundry.models.ops import grouped_gemm as gg
from llmfoundry.models.ops.float8.triton_kernels.utils import build_m_indices
from llmfoundry.models.parallel.tensor import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    _initialize_tp_weight,
    set_tensor_model_parallel_attributes,
    all_gather_linear,
    linear_reduce_scatter,
)

ORTHOGONAL_INIT_SCALE = 10.0


class GigaLoraModel(LoraModel):
    def __init__(
        self, model, config, adapter_name, model_cfg, low_cpu_mem_usage: bool = False
    ) -> None:
        self.config = model_cfg
        super().__init__(
            model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
        )

    def _create_new_module(self, lora_config, adapter_name, target, target_name, **kwargs):
        # for rank_pattern 
        config_params = lora_config.to_dict()
        config_params.update(kwargs)  
        # kwargs перезаписывает значения из config_params
        if target_name == 'qkv_proj':
            return LoraQKV(
                        base_layer=target,
                        adapter_name=adapter_name,
                        config=self.config,
                        **config_params,
                    )
        elif target_name == 'experts':
            return LoraGroupedMLP(
                base_layer=target,
                adapter_name=adapter_name,
                init_device=self.config.init_device,
                **config_params,
            )
        else:
            return LoraLinear(
                base_layer=target,
                adapter_name=adapter_name,
                init_device=self.config.init_device,
                **config_params,
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
        # --- split rank between uk_proj and uv_proj ---
        name = current_key.lower()

        is_uk = name.endswith(".uk_proj") or ".uk_proj." in name
        is_uv = name.endswith(".uv_proj") or ".uv_proj." in name
        is_dkv = name.endswith(".dkv_proj") or ".dkv_proj." in name
        is_kr = name.endswith(".kr_proj") or ".kr_proj." in name

        if is_uk or is_uv or is_dkv or is_kr:
            r_total = int(r)

            # If rank is too small, don't LoRA these layers at all.
            # (Otherwise you'd end up with uk=1 and uv=1 -> sum=2 > r_total=1)
            if r_total < 2:
                logger.warning(f"Skip LoRA for {current_key}: r_total={r_total} < 2 (can't split)")
                return  # do not replace this module with LoRA

            r_uk = r_total // 2
            r_uv = r_total - r_uk  # remainder goes to uv       
            r_dkv = r_total // 2
            r_kr = r_total - r_dkv  # remainder goes to kr

            r = r_uk if is_uk else r_uv if is_uv else r_dkv if is_dkv else r_kr if is_kr else 0
        # --- end split ---
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        # GroupedLlamaMLP (MoE experts) - always create new module
        if isinstance(target, GroupedLlamaMLP):
            new_module = self._create_new_module(lora_config, adapter_name, target, target_name, **kwargs)
            if new_module is None:
                return
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        elif isinstance(target, LoraGroupedMLP):
            # Already wrapped - skip (LoraGroupedMLP doesn't support multiple adapters yet)
            pass
        elif isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            if isinstance(target, LoraLinear):
                target._lora_target_name = current_key
            target.update_layer(
                adapter_name=adapter_name,
                r=r,
                init_device=self.config.init_device,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, target_name, **kwargs)
            if new_module is None:
                return
            if isinstance(new_module, LoraLinear):
                new_module._lora_target_name = current_key
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

class LoraModuleDict(nn.ModuleDict):
    """
    Специальный класс только для оборачивания лора модулей в gigafsdp
    """

    def __init__(self, *args, **kwargs):
        super(LoraModuleDict, self).__init__(*args, **kwargs)


class LoraLinear(nn.Module, LoraLayer):
    def __init__(
        self,
        init_device: str,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        r: int = 0,
        base_layer: Optional[Union[RowParallelLinear, ColumnParallelLinear, nn.Linear]] = None,
        adapter_name: str = 'default',
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, Literal["orthogonal", "gaussian", "xavier"]] = True,
        fan_in_fan_out: bool = False,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        
        super().__init__()

        LoraLayer.__init__(self, base_layer=base_layer)

        if base_layer is None:
            assert in_features is not None and out_features is not None, (
                "in LORA when 'base_layer is None' can not 'in_features is None or out_features is None' in same time"
            )
            self.in_features = in_features
            self.out_features = out_features
        else:
            if isinstance(base_layer, ColumnParallelLinear) or isinstance(base_layer, RowParallelLinear):
                self.in_features = self.base_layer._tp_linear_submodule.in_features
                self.out_features = self.base_layer._tp_linear_submodule.out_features
            else:
                assert isinstance(self.base_layer, nn.Linear), "LORA layers can use only with Linear"
                self.in_features = self.base_layer.in_features
                self.out_features = self.base_layer.out_features
                
        self.lora_A = LoraModuleDict()
        self.lora_B = LoraModuleDict()
        
        if use_dora:
            raise ValueError("DoRA is not supported")
        if lora_bias:
            raise ValueError("lora_bias is not supported")

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            init_device=init_device,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
        )

    def _init_rules(self, adapter_name: str = 'default'):
        def raise_init_error(*args, **kwargs):
            raise RuntimeError("This module must be inited by it's parent reset_parameters")
        lora_a = self.lora_A[adapter_name]
        lora_b = self.lora_B[adapter_name]
        if isinstance(lora_a, nn.Linear):
            lora_a.skip_init = True
            lora_a.reset_parameters = raise_init_error
        if isinstance(lora_b, nn.Linear):
            lora_b.skip_init = True
            lora_b.reset_parameters = raise_init_error

    def update_layer(
        self,
        adapter_name: str,
        init_device: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        init_lora_weights: Union[bool, Literal["orthogonal", "gaussian", "xavier"]] = True,
        lora_bias: bool = False,
        **kwargs,
    ):
        if r <= 0:
            raise ValueError("Lora rank r must be positive integer")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_dropout[adapter_name] = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )
        
        lora_a = nn.Linear(self.in_features, r, bias=False)
        lora_b = nn.Linear(r, self.out_features, bias=False)

        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        self.lora_bias[adapter_name] = lora_bias

        # Scaling and initialization
        self.scaling[adapter_name] = (
            lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r
        )
        self.init_lora_weights = init_lora_weights
        
        if init_device == "cpu":
            self.reset_parameters()

        
    def orthogonal_init(self, adapter_name: str = 'default'):
        if self.r[adapter_name] % 2 != 0:
            raise ValueError(f"Orthogonal initialization requires the LoRA rank to be even, got {self.r[adapter_name]} instead.")
        with torch.no_grad():
            X = torch.randn(self.r[adapter_name], self.r[adapter_name])
            Q, _ = torch.linalg.qr(X)
            q_odd = Q[0::2, :]  # Odd rows
            q_even = Q[1::2, :]  # Even rows
            
            lora_A = torch.randn(self.in_features, (self.r[adapter_name]) // 2).mm(q_odd).T / ORTHOGONAL_INIT_SCALE
            lora_B = torch.randn((self.r[adapter_name] // 2), self.out_features).T.mm(q_even) / ORTHOGONAL_INIT_SCALE
            lora_A_w = nn.Parameter(lora_A.contiguous())
            lora_B_w = nn.Parameter(lora_B.contiguous())
            self.lora_A[adapter_name].weight = lora_A_w
            self.lora_B[adapter_name].weight = lora_B_w

    def reset_parameters(self, adapter_name: str = 'default'):
        if self.init_lora_weights is False:
            return

        # Ранний выход если адаптер не существует
        if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            return
        # Получаем ссылки на модули
        lora_a = self.lora_A[adapter_name]
        lora_b = self.lora_B[adapter_name]
        if self.init_lora_weights == "orthogonal":
            self.orthogonal_init(adapter_name=adapter_name)
        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            # Определяем функцию инициализации для lora_A
            if self.init_lora_weights is True:
                init_fn = lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif self.init_lora_weights == "gaussian":
                init_fn = lambda w: nn.init.normal_(w, std=1 / self.r[adapter_name])
            elif self.init_lora_weights == "xavier":
                init_fn = lambda w: nn.init.xavier_uniform_(w, gain=1.0)
            else:
                raise ValueError(f"Unknown initialization plear set the init_lora_weights in lora config {self.init_lora_weights=}")

            # Инициализируем модули
            self._init_lora_module(module=lora_a, init_fn=init_fn)
            self._init_lora_module(module=lora_b, init_fn=nn.init.zeros_)

        self._init_lora_module(
            module=self.base_layer,
            init_fn=lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
        )

    def _init_lora_module(self, module: nn.Module, init_fn: Callable[[torch.Tensor], None]):
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        if self.base_layer:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = 0.0

        for adapter in self.active_adapters:
            if adapter not in self.lora_A:
                continue
            dropout = self.lora_dropout[adapter]
            scaling = self.scaling[adapter]
            lora_A = self.lora_A[adapter]
            lora_B = self.lora_B[adapter]

            if isinstance(self.base_layer, RowParallelLinear):
                if self.base_layer.input_is_parallel:
                    input_parallel = x
                else:
                    input_parallel = scatter_to_tensor_model_parallel_region(x)
                output_ = lora_B(lora_A(dropout(input_parallel)), *args, **kwargs) * scaling
                output = reduce_from_tensor_model_parallel_region(output_)
                result = output + result

            elif isinstance(self.base_layer, ColumnParallelLinear):
                input_ = copy_to_tensor_model_parallel_region(x)
                output_parallel =  lora_B(lora_A(dropout(input_)), *args, **kwargs) * scaling

                if self.base_layer.gather_output:
                    # All-gather across the partitions.
                    output = gather_from_tensor_model_parallel_region(output_parallel)
                else:
                    output = output_parallel
                result = output + result
            else:
                result = result + lora_B(lora_A(dropout(x)), *args, **kwargs) * scaling

        return result


class LoraQKV(nn.Module):
    def __init__(
        self,
        base_layer,
        adapter_name,
        config,
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        self.adapter_name = adapter_name
        assert hasattr(config, 'tp_size') and config.tp_size > 0, "For lora you need correct tp_size in model.config"
        assert hasattr(config, 'num_attention_heads') and config.num_attention_heads > 0, "For lora you need correct num_attention_heads in model.config"
        assert hasattr(config, 'num_key_value_heads') and config.num_key_value_heads > 0, "For lora you need correct num_key_value_heads in model.config"
        assert hasattr(config, 'hidden_size') and config.hidden_size > 0, "For lora you need correct hidden_size in model.config"

        kv_hidden_size = config.hidden_size // config.num_attention_heads * config.num_key_value_heads
        kv_hidden_size = kv_hidden_size // config.tp_size
        attention_size  = config.hidden_size // config.tp_size
        if config.tp_size > 1:
            in_features = self.base_layer._tp_linear_submodule.in_features
        else:
            in_features = self.base_layer.in_features

        # Создаем отдельные LoRA модули для q, k, v
        self.q_proj = LoraLinear(
            in_features=in_features,
            out_features=attention_size,
            init_device=config.init_device,
            adapter_name=adapter_name,
            **kwargs
        )
        self.k_proj = LoraLinear(
            in_features=in_features,
            out_features=kv_hidden_size,
            init_device=config.init_device,
            adapter_name=adapter_name,
            **kwargs
        )
        self.v_proj = LoraLinear(
            in_features=in_features,
            out_features=kv_hidden_size,
            init_device=config.init_device,
            adapter_name=adapter_name,
            **kwargs
        )

    def forward(self, x, *args, **kwargs):
        # Вычисляем q, k, v проекции
        result = self.base_layer(x, *args, **kwargs)
        lora_proj_args = {'logical_batch_size': kwargs['logical_batch_size']} if 'logical_batch_size' in kwargs else {}

        q_lora_hs = self.q_proj(x, **lora_proj_args)
        k_lora_hs = self.k_proj(x, **lora_proj_args)
        v_lora_hs = self.v_proj(x, **lora_proj_args)

        qkv_states_lora_scaled = torch.cat([q_lora_hs, k_lora_hs, v_lora_hs], dim=2)

        # Объединяем результаты

        return qkv_states_lora_scaled + result


class LoraGroupedMLP(nn.Module):
    """
    LoRA adapter for GroupedLlamaMLP (MoE experts).

    Applies LoRA to each expert using grouped matrix multiplication (GMM) for efficiency.

    Architecture:
    - Base layer: GroupedLlamaMLP with weight1 and weight2
    - LoRA for weight1: lora_A1 [E, r, H] -> lora_B1 [E, 2*D, r]
    - LoRA for weight2: lora_A2 [E, r, D] -> lora_B2 [E, H, r]

    Weights are stored transposed (NT layout) for fp8 compatibility on SM90
    (DeepGEMM requires NT layout for forward/backward). GMM calls use trans_b=True.

    where:
    - E = num_local_experts
    - H = hidden_size
    - D = intermediate_size
    - r = lora_rank
    """

    def __init__(
        self,
        init_device: str,
        base_layer: GroupedLlamaMLP,
        adapter_name: str = 'default',
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, Literal["orthogonal", "gaussian", "xavier"]] = True,
        use_rslora: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.config = base_layer.config
        self.num_local_experts = base_layer.num_local_experts
        self.ep_size = base_layer.ep_size
        self.hidden_size = base_layer.hidden_size
        self.intermediate_size = self.config.intermediate_size

        self._use_new_expert_weight_layout = base_layer._use_new_expert_weight_layout
        self._use_float8_grouped_gemm = base_layer._use_float8_grouped_gemm
        self._float8_wgrad_backend = base_layer._float8_wgrad_backend
        self._float8_triton_row2col = base_layer._float8_triton_row2col
        self._float8_sparse_fused_swiglu_quant = base_layer._float8_sparse_fused_swiglu_quant
        self._deep_ep_enabled = base_layer._deep_ep_enabled

        self.r = r
        self.lora_alpha = lora_alpha
        self.adapter_name = adapter_name
        self.init_lora_weights = init_lora_weights

        self.scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        # LoRA weights — always NT layout [E, r, H] / [E, 2*D, r] etc.,
        # independent of base weight layout. GMM calls use trans_b=True.
        self.lora_A1 = nn.Parameter(
            torch.empty(self.num_local_experts, r, self.hidden_size)
        )
        self.lora_B1 = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_size, r)
        )

        self.lora_A2 = nn.Parameter(
            torch.empty(self.num_local_experts, r, self.intermediate_size)
        )
        self.lora_B2 = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden_size, r)
        )

        if init_device == "cpu":
            self.reset_parameters()

    @property
    def handle1(self):
        return self.base_layer.handle1

    @property
    def handle2(self):
        return self.base_layer.handle2

    @property
    def weight1(self):
        return self.base_layer.weight1

    @property
    def weight2(self):
        return self.base_layer.weight2

    def orthogonal_init(self):
        if self.r % 2 != 0:
            raise ValueError(f"Orthogonal initialization requires the LoRA rank to be even, got {self.r} instead.")
        with torch.no_grad():
            for expert_id in range(self.num_local_experts):
                X = torch.randn(self.r, self.r)
                Q, _ = torch.linalg.qr(X)
                q_odd = Q[0::2, :]
                q_even = Q[1::2, :]

                self.lora_A1.data[expert_id] = (torch.randn(self.hidden_size, self.r // 2).mm(q_odd) / ORTHOGONAL_INIT_SCALE).T
                self.lora_B1.data[expert_id] = (q_even.T.mm(torch.randn(self.r // 2, 2 * self.intermediate_size)) / ORTHOGONAL_INIT_SCALE).T

                X = torch.randn(self.r, self.r)
                Q, _ = torch.linalg.qr(X)
                q_odd = Q[0::2, :]
                q_even = Q[1::2, :]

                self.lora_A2.data[expert_id] = (torch.randn(self.intermediate_size, self.r // 2).mm(q_odd) / ORTHOGONAL_INIT_SCALE).T
                self.lora_B2.data[expert_id] = (q_even.T.mm(torch.randn(self.r // 2, self.hidden_size)) / ORTHOGONAL_INIT_SCALE).T

    def reset_parameters(self):
        if self.init_lora_weights is False:
            return

        if self.init_lora_weights == "orthogonal":
            self.orthogonal_init()
        else:
            if self.init_lora_weights is True:
                init_fn = lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif self.init_lora_weights == "gaussian":
                init_fn = lambda w: nn.init.normal_(w, std=1 / self.r)
            elif self.init_lora_weights == "xavier":
                init_fn = lambda w: nn.init.xavier_uniform_(w, gain=1.0)
            else:
                raise ValueError(f"Unknown initialization, please set the init_lora_weights in lora config {self.init_lora_weights=}")

            for expert_id in range(self.num_local_experts):
                init_fn(self.lora_A1[expert_id])
                nn.init.zeros_(self.lora_B1[expert_id])
                init_fn(self.lora_A2[expert_id])
                nn.init.zeros_(self.lora_B2[expert_id])

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        tokens_per_expert_cpu: torch.Tensor,
        tokens_per_expert_list: list[int],
        permuted_probs: torch.Tensor,
        weights_quantized: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = permuted_local_hidden_states
        if x.nelement() == 0:
            return self.base_layer(
                x, tokens_per_expert, tokens_per_expert_cpu,
                tokens_per_expert_list, permuted_probs, weights_quantized,
            )

        if self.training and self._use_float8_grouped_gemm and (self.ep_size == 1 or self._deep_ep_enabled):
            return self._forward_fp8(
                x, tokens_per_expert_list, permuted_probs, weights_quantized,
            )

        if self._use_new_expert_weight_layout:
            w1 = self.base_layer.weight1.view(self.num_local_experts, -1, self.config.hidden_size)
            w2 = self.base_layer.weight2.view(self.num_local_experts, self.config.hidden_size, -1)
            trans_b = True
        else:
            w1 = self.base_layer.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.base_layer.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
            trans_b = False

        base_fc1 = gg.ops.gmm(x, w1, tokens_per_expert_cpu, trans_b=trans_b)
        dropped_input = self.lora_dropout(x)
        lora_fc1 = gg.ops.gmm(
            gg.ops.gmm(dropped_input, self.lora_A1, tokens_per_expert_cpu, trans_b=True),
            self.lora_B1, tokens_per_expert_cpu, trans_b=True,
        )

        fc1_combined = base_fc1 + lora_fc1 * self.scaling
        intermediate = fused_swiglu(fc1_combined, permuted_probs.unsqueeze(-1))

        base_fc2 = gg.ops.gmm(intermediate, w2, tokens_per_expert_cpu, trans_b=trans_b)
        lora_fc2 = gg.ops.gmm(
            gg.ops.gmm(self.lora_dropout(intermediate), self.lora_A2, tokens_per_expert_cpu, trans_b=True),
            self.lora_B2, tokens_per_expert_cpu, trans_b=True,
        )

        return base_fc2 + lora_fc2 * self.scaling

    def _forward_fp8(
        self,
        x: torch.Tensor,
        tokens_per_expert_list: list[int],
        permuted_probs: torch.Tensor,
        weights_quantized: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        assert self._use_new_expert_weight_layout, (
            "Old expert weight layout is not supported for float8 training."
        )
        w1, w2 = self.base_layer.weight1, self.base_layer.weight2

        groups_padded_sizes_list = [(m + 127) // 128 * 128 for m in tokens_per_expert_list]
        groups_padded_sizes_tensor = torch.tensor(
            groups_padded_sizes_list, dtype=torch.int32, device=x.device,
        )
        m_indices = build_m_indices(groups_padded_sizes_list, groups_padded_sizes_tensor)

        assert self.base_layer.handle1 is not None, "GroupedGemmFp8Wrapper is not available!"
        base_fc1 = self.base_layer.handle1.group_gemm(
            tensor_groups=x,
            weights=w1,
            weights_quantized=weights_quantized[0] if weights_quantized is not None else None,
            wgrad_backend=self._float8_wgrad_backend,
            tensor_groups_sizes=groups_padded_sizes_list,
            tensor_groups_sizes_padded=groups_padded_sizes_tensor,
            m_indices=m_indices,
            triton_row2col=self._float8_triton_row2col,
        )

        dropped_input = self.lora_dropout(x)
        lora_fc1 = gg.ops.gmm(
            gg.ops.gmm(dropped_input, self.lora_A1, groups_padded_sizes_tensor, trans_b=True),
            self.lora_B1, groups_padded_sizes_tensor, trans_b=True,
        )

        fc1_combined = base_fc1 + lora_fc1 * self.scaling
        assert not self.base_layer._float8_sparse_fused_swiglu_quant, (
            "Fused fp8 SwiGLU quantization is not supported with LoRA experts."
        )
        permuted_probs = permuted_probs.unsqueeze(-1)
        intermediate = fused_swiglu(fc1_combined, permuted_probs)

        assert self.base_layer.handle2 is not None, "GroupedGemmFp8Wrapper is not available!"
        base_fc2 = self.base_layer.handle2.group_gemm(
            tensor_groups=intermediate,
            weights=w2,
            weights_quantized=weights_quantized[1] if weights_quantized is not None else None,
            wgrad_backend=self._float8_wgrad_backend,
            tensor_groups_sizes=groups_padded_sizes_list,
            tensor_groups_sizes_padded=groups_padded_sizes_tensor,
            m_indices=m_indices,
            triton_row2col=self._float8_triton_row2col,
        )

        lora_fc2 = gg.ops.gmm(
            gg.ops.gmm(self.lora_dropout(intermediate), self.lora_A2, groups_padded_sizes_tensor, trans_b=True),
            self.lora_B2, groups_padded_sizes_tensor, trans_b=True,
        )

        return base_fc2 + lora_fc2 * self.scaling
