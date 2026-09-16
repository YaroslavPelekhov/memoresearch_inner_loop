"""GigaMix model configuration"""

import math
import warnings
from typing import Dict, List, Literal, Optional, Union

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

GIGAMIX_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

VALID_ATTENTION_TYPES = [
    "LlamaAttention",
    "LlamaPackedAttention",
    "LlamaPackedRingAttention",
    "LlamaLatentAttention",
]


class GigaMixConfig(PretrainedConfig):
    r"""

    ПАРАМЕТРЫ ИЗ https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json


    This is the configuration class to store the configuration of a [`GigaMixModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gigar-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Gigar model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_krope_heads (`int`, *optional*, defaults to None):
            Number of key heads for RoPE for each attention layer in MLA.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        tied_modules_groups (`list`, *optional*, defaults to `[[]]`):
            List of lists of module names to tie. Each inner list will be tied to a single weight tensor.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        gated_attention (`bool`, *optional*, defaults to `False`):
            Whether to enable gated attention for supported non-packed attention classes.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_routed_experts (`int`, *optional*, defaults to 8):
            Number of experts per Sparse MLP layer.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        moe_router_num_groups (`int`, *optional*, defaults to None):
            Number of groups to divide experts into for group-limited routing.
        moe_router_group_topk (`int`, *optional*, defaults to None):
            Number of selected groups for group-limited routing.
        init_type (`str`, *optional*, defaults to `"giga"`):
            Initialization type to use. Can be one of the following:

            - "giga": Default Giga init using `xavier_normal_` with `gain=1.0`,
            - "deepseek": DeepSeek init using `normal_` with `std=0.006`.

            Note: `init_type="dclm"` and `init_type="olmo3"` are not supported for GigaMix architecture yet.
        first_router_aux_loss_coef (float, optional, defaults to 0.001):
            Auxiliary loss coefficient to use only for the first MoE layers (see first_seq_aux_moe_layers).
        first_seq_aux_moe_layers (int, optional, defaults to 0):
            Number of initial MoE layers that should use first_router_aux_loss_coef instead of router_aux_loss_coef.
        norm_topk_prob (bool, optional, defaults to True):
            Whether to renormalize the selected top-k gate probabilities so that they sum to 1 per token.
        first_k_dense_replace (int, optional, defaults to None):
            Replace the first K MoE layers with dense MLP layers (no expert routing). Set None to disable.
        use_shared_expert_sigmoid (bool, optional, defaults to False):
            If True, apply an additional sigmoid gate to shared expert outputs before adding them to routed MoE output.
        use_mla_scaling_factor (bool, optional, defaults to False):
            If True, applies a LongCat-Flash-like scaling correction in `LlamaLatentAttention`:
            `alpha_q = sqrt(hidden_size / q_hidden_dim)` and
            `alpha_kv = sqrt(hidden_size / kv_hidden_dim)`.
        use_mtp (bool, optional, defaults to False):
            Enable Multi-Token Prediction (MTP) heads in addition to the main decoder.
        mtp_predictor_num (int, optional, defaults to 1):
            Number of MTP heads stacked on top of the decoder.
        mtp_loss_weight (float, optional, defaults to 0.3):
            Weight of the MTP loss added to the main language modeling loss.
        mtp_block_type (Literal["dense", "moe"], optional, defaults to "moe"):
            Selects which decoder-layer implementation is used inside the MTP block.
            This affects only the internal MTP branch and does not change the main decoder stack.
        aux_loss_free (bool, optional, defaults to False):
            Use a bias-free balancing scheme (router expert biases are updated online) instead of/alongside the standard auxiliary load-balancing loss.
        moe_router_bias_update_rate (float, optional, defaults to 0.0001):
            Per-step update rate (gamma) for expert router bias when aux_loss_free=True.
        moe_deepep_num_sms (int, optional, defaults to None):
            Number of SMs (streaming multiprocessors) to dedicate to DeepEP kernels.
        moe_enable_deepep (bool, optional, defaults to False):
            Enable DeepEP for efficient MoE token dispatch and combine.
        scoring_func (Literal["softmax","sigmoid"], optional, defaults to "softmax"):
            Scoring function used by the gate to produce expert scores before top-k selection.
        moe_router_routed_scaling_factor (float, optional, defaults to 1.0):
            Multiplicative scaling factor applied to the routed expert weights/probabilities.
        aux_loss_free_strategy (Literal["baseline","indicator","threshold","adaptive"], optional, defaults to "baseline"):
            Strategy for bias-free balancing when aux_loss_free=True:
            baseline: step ∝ sign(avg - load);
            indicator: positive step if load < avg, else 0;
            threshold: positive if load < avg, negative if load > avg·(1+β);
            adaptive: like threshold, but step is additionally scaled by max-violation.
        adaptive_beta (float, optional, defaults to 0.0):
            Threshold margin β for threshold/adaptive strategies; experts above avg·(1+β) are considered overloaded.
        adaptive_alpha (float, optional, defaults to 1.0):
            Sensitivity of the adaptive strategy; scales the step with the max-violation.
        adaptive_scale (float, optional, defaults to 5.0):
            Maximum cap for the adaptive scaling factor to avoid instability.
        lm_head_logit_softcapping (float, optional, defaults to None):
            Value for soft clipping LM-head logits. Must be > 0 when set. When set to None, logits doesn't clip.
            DO NOT USE with use_liger=True.
        layernorm_type (str, optional, defaults to "pre"):
            Controls which layer normalization positions are applied in each decoder block. Must be one of:

            - "pre": Pre-norm only (Llama style). Applies `input_layernorm` before attention and
              `post_attention_layernorm` before the FFN/MoE. Residual is added after each sub-block output.
            - "post": Post-norm only (OLMo 3 style). Applies `post_self_attn_layernorm` after attention output and
              `post_feedforward_layernorm` after FFN/MoE output, both before the residual addition.
            - "pre_post": Both pre- and post-norm (Gemma 2 style). Applies all four norms: `input_layernorm`
              and `post_attention_layernorm` before each sub-block, and `post_self_attn_layernorm` and
              `post_feedforward_layernorm` after each sub-block output before the residual addition.
        linear_attention_type (Literal["Qwen3NextGatedDeltaNet", "KimiDeltaAttention"], *optional*, defaults to None):
            Set to None to disable hybrid attention, to enable a hybrid model, set this to one of: ["Qwen3NextGatedDeltaNet", "KimiDeltaAttention"]
        full_attention_layers (List[`int`], defaults to []):
            Indices of transformer layers that always use full attention, [0, ..., num_layers - 1]
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        linear_sp_implementation (`str`, *optional*, defaults to "all2all"):
            SP implementation for linear attention layers.
        linear_use_legacy_qkvz_layout (`bool`, *optional*, defaults to False):
            If True, the Qwen3NextGatedDeltaNet uses the legacy weight layout with a single
            fused `qkvz_proj` projection and three separate `{q,k,v}_conv1d` convolutions.
            Set True only for backward compatibility when loading checkpoints trained before
            the split into `qkv_proj` / `z_proj` and the fused `qkv_conv1d`.
        linear_gating_type (Literal["gated_rmsnorm", "gated_rmsnorm_sigmoid", "gated_rmsnorm_sigmoid_zero_centered", "sigmoid_gate"], *optional*, defaults to "gated_rmsnorm"):
            Gating/normalization applied to the Qwen3NextGatedDeltaNet output.
            `gated_rmsnorm_sigmoid_zero_centered` is an explicit alias for
            `FusedRMSNormGated(activation="sigmoid", zero_centered=True)` introduced
            for configs that want the zero-centered `1 + weight` parameterization
            to be spelled out at the gating-type level.
        swiglu_limit (float, optional, defaults to 0.0):
            DeepSeek-style activation clipping bound for fused SwiGLU (MoE and dense paths,
            both fp8 and non-fp8).
            gate: min(gate, limit), up: clamp(up, -limit, limit). 0.0 disables clamping (default).
        embed_scale (float, optional, defaults to 1.0):
            Scale applied to token embeddings output (now only for non-parallel nn.Embedding path).
        use_embed_ln (bool, optional, defaults to False):
            Apply LayerNorm to embeddings or not.
    """

    model_type = "giga_mix"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_krope_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        use_mla_scaling_factor: bool = False,
        apply_torch_compile_to_projections: bool = False,
        use_custom_rotary_kernel: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = False,  # it is useful only for inference
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        tied_modules_groups: Optional[List[List[str]]] = None,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Union[str, float]]] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        activation_checkpoint_layers_num: Optional[int] = None,
        num_experts_per_tok: int = 2,
        num_routed_experts: int = 8,
        router_aux_loss_coef: float = 0.001,
        first_router_aux_loss_coef: float = 0.001,
        first_seq_aux_moe_layers: int = 0,
        router_seq_aux: bool = True,
        router_legacy_weight_init: bool = False,
        fused_mlp: bool = False,
        fused_mlp_checkpoint_lvl: int = 3,
        moe_impl: Literal[
            "DeepseekGMMMoeBlock",
            "SonicGMMMoeBlock",
            "ScMoEBlock",
        ] = "DeepseekGMMMoeBlock",
        norm_type: Literal[
            "rmsnorm",
            "LlamaRMSNorm",
            "ZeroCenteredRMSNorm",
            "ZeroCenteredGatedNorm",
        ] = "LlamaRMSNorm",
        norm_topk_prob: bool = True,
        # DeepSeek
        n_shared_experts: Optional[int] = None,
        first_k_dense_replace: Optional[int] = None,
        use_shared_expert_sigmoid: bool = False,
        dense_intermediate_size: Optional[int] = None,
        dense_fused_mlp_checkpoint_lvl: int = 2,
        init_device: Optional[Union[str, torch.device]] = "cpu",
        loss_inplace_backward: bool = False,
        tp_size: Optional[int] = None,
        attention_hidden_size: Optional[int] = None,
        attention_type: Literal[
            "LlamaAttention",
            "LlamaPackedAttention",
            "LlamaPackedRingAttention",
            "LlamaLatentAttention",
            "Qwen3NextGatedDeltaNet",
            "KimiDeltaAttention",
        ] = "LlamaAttention",
        gating_type: Literal[
            "top-k", "dummy_uniform", "first_expert_group", "dynamic"
        ] = "top-k",
        first_group_steps: int = 0,
        deterministic_attention: bool = False,
        sp_split_type: Optional[Literal["equal", "zigzag"]] = None,
        use_liger: bool = False,
        delete_logits: bool = False,
        skip_init_tp_modules: bool = True,
        use_cache_force: bool = False,
        z_loss_eps: float = 0.0,
        hidden_z_loss_eps: float = 0.0,
        hidden_z_loss_attn_coef: float = 0.0,
        hidden_z_loss_moe_coef: float = 0.0,
        hidden_z_loss_dense_mlp_coef: float = 0.0,
        router_z_loss_eps: float = 0.0,
        use_mtp: bool = False,
        mtp_predictor_num: int = 1,
        mtp_loss_weight: float = 0.3,
        mtp_block_type: Literal["dense", "moe"] = "moe",
        ignore_index: int = -100,
        apply_qk_norm: bool = False,
        llama3_ring_heads_k_stride: int = 1,
        enable_async_tp: bool = False,
        parallel_embedding_type: Literal[
            "VocabParallelEmbedding",
            "EmbeddingParallelEmbedding",
        ] = "EmbeddingParallelEmbedding",
        aux_loss_free: bool = False,
        moe_router_bias_update_rate: float = 0.0001,
        moe_router_num_groups: Optional[int] = None,
        moe_router_group_topk: Optional[int] = None,
        dispatcher_balancing_strategy: Optional[
            Literal["capacity_factor", "fixed_capacity", "peak_capacity_factor"]
        ] = None,
        expert_capacity_factor: Optional[float] = None,
        expert_capacity_tokens: Optional[int] = None,
        drop_by_group_capacity: bool = False,
        peak_capacity_microbatchsize: Optional[int] = None,
        moe_deepep_num_sms: Optional[int] = None,
        moe_enable_deepep: bool = False,
        scmoe_save_dispatch_for_backward: bool = False,
        scoring_func: Literal["softmax", "sigmoid"] = "softmax",
        moe_router_routed_scaling_factor: float = 1.0,
        aux_loss_free_strategy: Literal[
            "baseline", "indicator", "threshold", "adaptive"
        ] = "baseline",
        adaptive_beta: float = 0.0,
        adaptive_alpha: float = 1.0,
        adaptive_scale: float = 5.0,
        rope_origin: Literal["custom", "deepseek"] = "custom",
        init_type: Literal["giga", "deepseek", "dclm", "olmo3"] = "giga",
        deepseek_init_std: Optional[float] = 0.006,
        use_new_expert_weight_layout: bool = False,
        start_from_sf_checkpoint_in_old_expert_weight_layout: bool = False,
        # float8 grouped gemm (MoE / sparse experts, blockwise quant)
        use_float8_grouped_gemm: bool = False,
        float8_wgrad_backend: Literal["deep_gemm", "transformer_engine"] = "deep_gemm",
        float8_deep_gemm_num_sms: int = 116,
        float8_triton_row2col: bool = False,
        float8_blocking_weight_quant: bool = False,
        float8_sparse_fused_swiglu_quant: bool = True,
        float8_quantize_at_boundary: bool = False,
        swiglu_limit: float = 0.0,     # 0.0 = disabled; applies to MoE (fp8 + non-fp8) and dense (fp8 + non-fp8) paths
        # float8 dense MLP (TE delayed scaling + optional swiglu_limit clamp)
        float8_dense_fused_swiglu_quant: bool = False,
        # end float8 arguments
        balance_without_prefix: Optional[bool] = False,
        renorm_router_weights: Optional[bool] = False,
        gated_attention: Optional[bool] = False,
        lm_head_logit_softcapping: Optional[float] = None,
        layernorm_type: Literal["pre", "post", "pre_post"] = "pre",
        linear_attention_type: Optional[
            Literal["Qwen3NextGatedDeltaNet", "KimiDeltaAttention"]
        ] = None,
        full_attention_layers: Optional[list[int]] = None,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_sp_implementation: str = "all2all",
        linear_gating_type: Literal["gated_rmsnorm", "gated_rmsnorm_sigmoid", "gated_rmsnorm_sigmoid_zero_centered", "sigmoid_gate"] = "gated_rmsnorm",
        linear_sigmoid_gate_scale: float = 1.0,
        linear_attn_o_norm_eps: Optional[float] = None,
        linear_conv_init_std: float = 0.006,
        linear_use_legacy_qkvz_layout: bool = False,
        embed_scale: float = 1.0,
        use_embed_ln: bool = False,
        layernorm_gating_weight: float = 2.0,
        use_master_weight: bool = True,
        **kwargs,
    ):
        if "mega" in moe_impl.lower():
            assert intermediate_size % 128 == 0, (
                "With Megablocks intermediate size must be divisible by 128"
            )
            assert hidden_size % 128 == 0, (
                "With Megablocks hidden size must be divisible by 128"
            )

        if n_shared_experts is not None:
            moe_impl_lower = moe_impl.lower()
            assert "deepseek" in moe_impl_lower or moe_impl_lower == "scmoeblock", (
                "With n_shared_experts, use a DeepSeek-style MoE block (name containing "
                "'deepseek') or ScMoEBlock; shared experts are wired outside the MoE block."
            )
            assert num_experts_per_tok > 0, (
                f"{n_shared_experts=} {num_experts_per_tok=}"
            )
        if use_shared_expert_sigmoid and (
            n_shared_experts is None or n_shared_experts <= 0
        ):
            raise AssertionError(
                "`use_shared_expert_sigmoid=True` requires `n_shared_experts > 0`."
            )
        if mtp_block_type not in {"dense", "moe"}:
            raise ValueError(
                f"mtp_block_type must be one of ('dense', 'moe'), got '{mtp_block_type}'."
            )

        normalized_full_attention_layers = full_attention_layers or []
        mtp_attention_anchor_layer_idx = max((num_hidden_layers or 0) - 1, 0)

        if (
            use_mtp
            and mtp_predictor_num > 0
            and mtp_block_type == "dense"
            and linear_attention_type is not None
            and mtp_attention_anchor_layer_idx not in normalized_full_attention_layers
        ):
            raise ValueError(
                "Dense MTP requires full attention on its anchor layer. "
                f"Add layer index {mtp_attention_anchor_layer_idx} to "
                "`full_attention_layers`, disable hybrid attention, "
                "or use `mtp_block_type='moe'`."
            )

        self.moe_impl = moe_impl

        if router_legacy_weight_init:
            warnings.warn(
                "Legacy router_legacy_weight_init is set to True, "
                + "please use router_legacy_weight_init=False in new production pretrains"
            )
        self.router_legacy_weight_init = router_legacy_weight_init

        self.dense_fused_mlp_checkpoint_lvl = dense_fused_mlp_checkpoint_lvl
        self.first_k_dense_replace = first_k_dense_replace
        self.dense_intermediate_size = dense_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.use_shared_expert_sigmoid = use_shared_expert_sigmoid
        self.norm_topk_prob = norm_topk_prob
        self.router_seq_aux = router_seq_aux

        assert not fused_mlp, "Fused MLP is banned for MoE"
        self.fused_mlp = fused_mlp
        self.fused_mlp_checkpoint_lvl = fused_mlp_checkpoint_lvl

        self.gating_type = gating_type
        self.first_group_steps = first_group_steps

        self.scoring_func = scoring_func
        self.moe_router_routed_scaling_factor = moe_router_routed_scaling_factor
        self.router_aux_loss_coef = router_aux_loss_coef
        self.first_router_aux_loss_coef = first_router_aux_loss_coef
        self.first_seq_aux_moe_layers = first_seq_aux_moe_layers
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.activation_checkpoint_layers_num = activation_checkpoint_layers_num

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_krope_heads = num_krope_heads

        self.num_key_value_heads = num_key_value_heads

        if (
            qk_nope_head_dim
            or qk_rope_head_dim
            or v_head_dim
            or kv_lora_rank
            or q_lora_rank
            or use_mla_scaling_factor
        ):
            assert attention_type == "LlamaLatentAttention", (
                f"Parameters for MLA are set, but attention type is {attention_type}"
            )

        self.head_dim = head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank

        self.use_mla_scaling_factor = use_mla_scaling_factor
        self.apply_torch_compile_to_projections = apply_torch_compile_to_projections
        self.use_custom_rotary_kernel = use_custom_rotary_kernel

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        # ignore this flag at the moment
        self.use_cache = use_cache
        # but kv cache enable in audio with packed attn and gqa, thus, we add this flag
        # to force use cache and don't change original use_cache
        self.use_cache_force = use_cache_force
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        if isinstance(self.rope_scaling, DictConfig):
            self.rope_scaling = om.to_container(self.rope_scaling, resolve=True)

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if attention_type not in VALID_ATTENTION_TYPES:
            raise ValueError(
                f"Got invalid `attention_type` option! The only supported options are {VALID_ATTENTION_TYPES}, but got {attention_type}."
            )
        self.attention_type = attention_type
        self.deterministic_attention = deterministic_attention
        self.init_device = init_device

        self.tp_size = tp_size
        self.enable_async_tp = enable_async_tp
        if self.enable_async_tp:
            assert not fused_mlp, "Fused MLP IS NOT supported for async TP"

        self.parallel_embedding_type = parallel_embedding_type
        # print(f"{tp_size=} {enable_async_tp=}")
        # assert self.tp_size is not None and self.tp_size > 1 and self.enable_async_tp, \
        #     "Tensor parallelism is supported only with async TP"

        self.tp_group = None
        self.loss_inplace_backward = loss_inplace_backward

        self.sp_split_type = sp_split_type

        self.attention_hidden_size = attention_hidden_size

        self.use_liger = use_liger
        self.delete_logits = delete_logits

        if lm_head_logit_softcapping is not None and lm_head_logit_softcapping <= 0:
            raise ValueError(
                f"lm_head_logit_softcapping must be > 0 when provided, got {lm_head_logit_softcapping}."
            )
        self.lm_head_logit_softcapping = lm_head_logit_softcapping

        self.z_loss_eps = z_loss_eps
        self.hidden_z_loss_eps = hidden_z_loss_eps
        self.hidden_z_loss_attn_coef = hidden_z_loss_attn_coef
        self.hidden_z_loss_moe_coef = hidden_z_loss_moe_coef
        self.hidden_z_loss_dense_mlp_coef = hidden_z_loss_dense_mlp_coef
        self.router_z_loss_eps = router_z_loss_eps
        self.apply_qk_norm = apply_qk_norm

        self.skip_init_tp_modules = skip_init_tp_modules

        self.llama3_ring_heads_k_stride = llama3_ring_heads_k_stride

        self.aux_loss_free = aux_loss_free
        self.moe_router_bias_update_rate = moe_router_bias_update_rate

        self.moe_router_num_groups = moe_router_num_groups
        self.moe_router_group_topk = moe_router_group_topk

        self.use_mtp = use_mtp and mtp_predictor_num > 0
        self.mtp_predictor_num = mtp_predictor_num
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_block_type = mtp_block_type

        self.ignore_index = ignore_index

        self.norm_type = norm_type

        # token drop parameters
        self.dispatcher_balancing_strategy = dispatcher_balancing_strategy
        self.expert_capacity_factor = expert_capacity_factor
        self.expert_capacity_tokens = expert_capacity_tokens
        self.drop_by_group_capacity = drop_by_group_capacity
        self.peak_capacity_microbatchsize = peak_capacity_microbatchsize

        self.moe_enable_deepep = moe_enable_deepep
        self.moe_deepep_num_sms = moe_deepep_num_sms
        self.scmoe_save_dispatch_for_backward = scmoe_save_dispatch_for_backward

        self.aux_loss_free_strategy = aux_loss_free_strategy
        self.adaptive_beta = adaptive_beta
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_scale = adaptive_scale

        self.rope_origin = rope_origin

        if init_type in {"dclm", "olmo3"}:
            raise ValueError(
                f'Init type "{init_type}" is not supported for MoE architecture yet.'
            )
        self.init_type = init_type
        self.deepseek_init_std = deepseek_init_std

        # sparse float8 training
        self.use_new_expert_weight_layout = use_new_expert_weight_layout
        self.start_from_sf_checkpoint_in_old_expert_weight_layout = (
            start_from_sf_checkpoint_in_old_expert_weight_layout
        )
        if self.use_new_expert_weight_layout:
            warnings.warn("Use new expert weight layout.")

        assert float8_wgrad_backend in {"deep_gemm", "transformer_engine"}, (
            f"float8_wgrad_backend must be 'deep_gemm' or 'transformer_engine', got {float8_wgrad_backend!r}"
        )
        self.use_float8_grouped_gemm = use_float8_grouped_gemm
        self.float8_wgrad_backend = float8_wgrad_backend
        self.float8_triton_row2col = float8_triton_row2col
        self.float8_deep_gemm_num_sms = float8_deep_gemm_num_sms
        self.float8_blocking_weight_quant = float8_blocking_weight_quant
        self.float8_sparse_fused_swiglu_quant = float8_sparse_fused_swiglu_quant
        self.float8_quantize_at_boundary = float8_quantize_at_boundary
        self.swiglu_limit = swiglu_limit
        if swiglu_limit > 0:
            assert tp_size == 1 or tp_size is None, (
                "swiglu_limit>0 requires tp_size=1: SwiGLU activation clipping is "
                "only implemented for LlamaMLP, not for ParallelMLP. "
                f"Got swiglu_limit={swiglu_limit}, tp_size={tp_size}."
            )
        self.float8_dense_fused_swiglu_quant = float8_dense_fused_swiglu_quant

        self.tied_modules_groups = (
            tied_modules_groups if tied_modules_groups is not None else [[]]
        )
        self.linear_attention_type = linear_attention_type
        assert self.linear_attention_type != "KimiDeltaAttention", "Not tested yet"

        self.full_attention_layers = normalized_full_attention_layers
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_sp_implementation = linear_sp_implementation
        assert self.linear_sp_implementation == "all2all", (
            "Only all2all sequence parallelism is supported for linear attention"
        )
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_conv_init_std = linear_conv_init_std
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout

        self.balance_without_prefix = balance_without_prefix

        self.renorm_router_weights = renorm_router_weights
        self.gated_attention = gated_attention
        if self.gated_attention and self.pretraining_tp > 1:
            raise AssertionError(
                "`gated_attention` is not supported with `pretraining_tp > 1`."
            )
        if self.gated_attention and attention_type in {
            "LlamaPackedAttention",
            "LlamaPackedRingAttention",
        }:
            raise AssertionError(
                "`gated_attention` is not supported for packed or ring attention classes."
            )

        assert layernorm_type in {"pre", "post", "pre_post"}, (
            f"Invalid layernorm_type: {layernorm_type}. Must be one of: pre, post, pre_post."
        )
        self.layernorm_type = layernorm_type

        if (
            hidden_z_loss_attn_coef != 0.0
            or hidden_z_loss_moe_coef != 0.0
            or hidden_z_loss_dense_mlp_coef != 0.0
        ) and layernorm_type == "pre":
            raise ValueError(
                "hidden_z_loss_attn_coef/hidden_z_loss_moe_coef/hidden_z_loss_dense_mlp_coef "
                "require layernorm_type in {'post', 'pre_post'} (the hook attaches before "
                "post-norm activations which do not exist with layernorm_type='pre'), "
                f"got layernorm_type='{layernorm_type}'."
            )

        if hidden_z_loss_dense_mlp_coef != 0.0 and not first_k_dense_replace:
            raise ValueError(
                "hidden_z_loss_dense_mlp_coef has no effect unless first_k_dense_replace > 0; "
                f"got first_k_dense_replace={first_k_dense_replace}."
            )
        self.linear_sp_implementation = linear_sp_implementation

        if not math.isclose(embed_scale, 1.0, rel_tol=0.0, abs_tol=1e-8) and (
            tp_size is not None and tp_size > 1
        ):
            raise ValueError(
                "Embedding scalling does not support Parallel Embeddings yet"
            )
        self.embed_scale = embed_scale

        self.use_embed_ln = use_embed_ln

        self.layernorm_gating_weight = layernorm_gating_weight
        self.use_master_weight = use_master_weight

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
