"""Gigar model configuration"""

from typing import Dict, List, Literal, Optional, Union

import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

from omegaconf import DictConfig
from omegaconf import OmegaConf as om


GIGAR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

VALID_ATTENTION_TYPES = [
    "LlamaAttention",
    "LlamaPackedAttention",
    "LlamaPackedRingAttention",
    "LlamaLatentAttention",
]


class GigarConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GigarModel`]. It is used to instantiate an LLaMA
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
        init_type (`str`, *optional*, defaults to `"giga"`):
            Initialization type to use. Can be one of the following:
            - "giga": Default Giga init using `xavier_normal_` with `gain=1.0`,
            - "deepseek": DeepSeek init using `normal_` with `std=0.006`,
            - "dclm": DCLM style init using `tranc_normal_` with 3 sigma boundaries and depth scaling (`std = 1/sqrt(fan_in)` (`hidden_size` or `intermediate_size`), `1/sqrt(2*(layer_idx+1))` depth scaling for output projections).
            - "olmo3": OLMo3/nGPT-style normalized init: `normal_` (std=hidden_size**-0.5) for embeddings, `trunc_normal_` (std=hidden_size**-0.5) for linears with number of layers depth scaling.
        layernorm_type (str, optional, defaults to "pre"):
            Controls which layer normalization positions are applied in each decoder block. Must be one of:

            - "pre": Pre-norm only (Llama style). Applies `input_layernorm` before attention and
              `post_attention_layernorm` before the FFN. Residual is added after each sub-block output.
            - "post": Post-norm only (OLMo 3 style). Applies `post_self_attn_layernorm` after attention output and
              `post_feedforward_layernorm` after FFN output, both before the residual addition.
            - "pre_post": Both pre- and post-norm (Gemma 2 style). Applies all four norms: `input_layernorm`
              and `post_attention_layernorm` before each sub-block, and `post_self_attn_layernorm` and
              `post_feedforward_layernorm` after each sub-block output before the residual addition.
            - "deepseek": DeepSeek init using `normal_` with `std=0.006`.
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

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gigar"
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
        fused_mlp: bool = False,
        float8_dense_fused_swiglu_quant: bool = False,
        fused_mlp_checkpoint_lvl: int = 0,
        tp_size: Optional[int] = None,
        init_device: Optional[Union[str, torch.device]] = "cpu",
        loss_inplace_backward: bool = False,
        attention_hidden_size: Optional[int] = None,
        attention_type: Literal[
            "LlamaAttention",
            "LlamaPackedAttention",
            "LlamaPackedRingAttention",
            "LlamaLatentAttention",
            "Qwen3NextGatedDeltaNet",
            "KimiDeltaAttention"
        ] = "LlamaAttention",
        norm_type: Literal["LlamaRMSNorm", "rmsnorm"] = "LlamaRMSNorm",
        parallel_embedding_type: Literal[
            "VocabParallelEmbedding",
            "EmbeddingParallelEmbedding",
        ] = "EmbeddingParallelEmbedding",
        deterministic_attention: bool = False,
        gated_attention: bool = False,
        sp_split_type: Optional[Literal["equal", "zigzag"]] = None,
        freeze_non_embed: bool = False,
        non_freeze_layers_idxs: Optional[list] = None,
        enable_async_tp: bool = False,
        use_liger: bool = False,
        delete_logits: bool = False,
        use_mtp: bool = False,
        mtp_predictor_num: int = 1,
        mtp_loss_weight: float = 0.3,
        skip_init_tp_modules: bool = True,
        use_cache_force: bool = False,
        z_loss_eps: float = 0.0,
        ignore_index: int = -100,
        apply_qk_norm: bool = False,
        llama3_ring_heads_k_stride: int = 1,
        rope_origin: Literal["custom", "deepseek"] = "custom",
        init_type: Literal["giga", "deepseek", "dclm", "olmo3"] = "giga",
        layernorm_type: Literal["pre", "post", "pre_post"] = "pre",
        linear_attention_type: Optional[Literal["Qwen3NextGatedDeltaNet", "KimiDeltaAttention"]] = None,
        full_attention_layers: Optional[list[int]] = None,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_sp_implementation: str = "all2all",
        layernorm_gating_weight: float = 2,
        use_master_weight: bool = True,
        decoder_layer_module: Optional[str] = None,
        decoder_layer_type: str = "LlamaDecoderLayer",
        decoder_layer_kwargs: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        self.activation_checkpoint_layers_num = activation_checkpoint_layers_num

        self.fused_mlp = fused_mlp
        self.fused_mlp_checkpoint_lvl = fused_mlp_checkpoint_lvl
        self.loss_inplace_backward = loss_inplace_backward

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

        self.float8_dense_fused_swiglu_quant = float8_dense_fused_swiglu_quant
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
        self.enable_async_tp = enable_async_tp

        if attention_type not in VALID_ATTENTION_TYPES:
            raise ValueError(
                f"Got invalid `attention_type` option! The only supported options are {VALID_ATTENTION_TYPES}, but got {attention_type}."
            )
        self.attention_type = attention_type
        self.deterministic_attention = deterministic_attention
        self.norm_type = norm_type

        self.parallel_embedding_type = parallel_embedding_type

        self.init_device = init_device

        self.tp_size = tp_size
        self.tp_group = None

        self.sp_split_type = sp_split_type
        self.attention_hidden_size = attention_hidden_size

        self.freeze_non_embed = freeze_non_embed

        self.non_freeze_layers_idxs = non_freeze_layers_idxs
        if self.non_freeze_layers_idxs is not None:
            assert not freeze_non_embed
            assert type(self.non_freeze_layers_idxs) == list
            assert len(self.non_freeze_layers_idxs) > 0

        self.use_liger = use_liger
        self.delete_logits = delete_logits

        # Multi-Token Prediction
        self.use_mtp = use_mtp and mtp_predictor_num > 0
        self.mtp_predictor_num = mtp_predictor_num
        self.mtp_loss_weight = mtp_loss_weight

        self.skip_init_tp_modules = skip_init_tp_modules

        self.z_loss_eps = z_loss_eps
        self.apply_qk_norm = apply_qk_norm

        self.llama3_ring_heads_k_stride = llama3_ring_heads_k_stride

        self.ignore_index = ignore_index

        self.rope_origin = rope_origin

        self.init_type = init_type

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

        self.tied_modules_groups = (
            tied_modules_groups if tied_modules_groups is not None else [[]]
        )
        assert layernorm_type in {"pre", "post", "pre_post"}, (
            f"Invalid layernorm_type: {layernorm_type}. Must be one of: pre, post, pre_post."
        )
        self.layernorm_type = layernorm_type
        self.linear_attention_type = linear_attention_type
        assert self.linear_attention_type is None, "Linear Attention is not supported for dense models yet"
        assert self.linear_attention_type != "KimiDeltaAttention", "Not tested yet"

        self.full_attention_layers = full_attention_layers or []
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_sp_implementation = linear_sp_implementation
        assert self.linear_sp_implementation == "all2all", "Only all2all sequence parallelism is supported for linear attention"

        self.layernorm_gating_weight = layernorm_gating_weight
        self.use_master_weight = use_master_weight

        # Optional external decoder plugin. The defaults preserve the normal
        # LLM Foundry construction path. Autoresearch configurations can import
        # a module that registers a candidate block and pass candidate-specific
        # constructor arguments without changing GigarModel.
        self.decoder_layer_module = decoder_layer_module
        self.decoder_layer_type = decoder_layer_type
        # Keep the unset value as None so Composer's config override logic can
        # accept an arbitrary mapping supplied by a candidate config.
        self.decoder_layer_kwargs = (
            None if decoder_layer_kwargs is None else dict(decoder_layer_kwargs)
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def non_freeze_layers_idxs(self) -> Optional[list]:
        return self._non_freeze_layers_idxs

    @non_freeze_layers_idxs.setter
    def non_freeze_layers_idxs(self, value) -> Optional[list]:
        self._non_freeze_layers_idxs = list(value) if value is not None else None
        if self._non_freeze_layers_idxs is not None:
            assert not self.freeze_non_embed
