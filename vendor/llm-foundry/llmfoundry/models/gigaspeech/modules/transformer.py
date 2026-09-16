import typing as tp

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from llmfoundry.models.gigaspeech.modules.conformer import (
    ConformerConfig,
    StackingSubsampler,
    ConvSubsampling,
    ChunkRotaryPositionMultiHeadedAttention,
    ChunkRotaryPositionalEmbedding,
)
from llmfoundry.models.layers import FFN_CLASS_REGISTRY
from llmfoundry.models.layers.norm import resolve_norm_class


class TransformerEncoderConfig(PretrainedConfig):
    model_type: str = "transformer_encoder"

    def __init__(
            self,
            feat_in: int = 64,
            d_model: int = 1024,
            intermediate_size: int = 4096,
            n_heads: int = 16,
            self_attention_model: str = "chunk_rotary",
            max_position_seq_len: int = 131072,
            pos_emb_max_len: int = 131072,
            attn_chunk_size: int = 200,
            subsampling: str = "striding_conv1d",
            subsampling_factor: int = 4,
            subsampling_conv_channels: int = 1024,
            num_hidden_layers: int = 32,
            hidden_size: int = 1024,
            rms_norm_eps: float = 1e-6,  
            init_device: str = "cpu",
            dropout_att: float = 0.0,
            norm_type: str = "rmsnorm",
            tp_size: tp.Optional[int] = None,
            hidden_act: str = "silu",
            fused_mlp: bool = False,
            fused_mlp_checkpoint_lvl: int = 0,
            pretraining_tp: int = 1,
            **kwargs: tp.Dict
        ) -> None:
            super().__init__(**kwargs)

            self.feat_in = feat_in
            self.d_model = d_model
            self.intermediate_size = intermediate_size
            self.n_heads = n_heads
            self.self_attention_model = self_attention_model
            self.max_position_seq_len = max_position_seq_len
            self.pos_emb_max_len = pos_emb_max_len
            self.attn_chunk_size = attn_chunk_size
            self.subsampling = subsampling
            self.subsampling_factor = subsampling_factor
            self.subsampling_conv_channels = subsampling_conv_channels
            self.num_hidden_layers = num_hidden_layers
            self.hidden_size = hidden_size
            self.rms_norm_eps = rms_norm_eps
            self.init_device = init_device
            self.dropout_att = dropout_att
            self.norm_type = norm_type
            self.tp_size = tp_size
            self.hidden_act = hidden_act
            self.fused_mlp = fused_mlp
            self.fused_mlp_checkpoint_lvl = fused_mlp_checkpoint_lvl
            self.pretraining_tp = pretraining_tp


class LlamaDecoderLayerSpeech(torch.nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs: tp.Dict) -> None:
        super().__init__(**kwargs)
        self._config = config

        self.hidden_size = config.hidden_size

        assert config.self_attention_model == "chunk_rotary"
        self.self_attn = ChunkRotaryPositionMultiHeadedAttention(
            n_head=self._config.n_heads,
            n_feat=self._config.d_model,
            dropout=self._config.dropout_att,
            attn_chunk_size=self._config.attn_chunk_size,
        )

        norm_class = resolve_norm_class(config.norm_type)
        self.input_layernorm = norm_class(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.init_device
        )
        self.post_attention_layernorm = norm_class(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.init_device
        )
        assert config.tp_size == 1 or config.tp_size is None
        self.mlp = FFN_CLASS_REGISTRY["LlamaMLP"](config)

        assert not config.fused_mlp
        self._is_fused_mlp = False

    def forward(self, audios: torch.Tensor, pos_emb: tp.Optional[torch.Tensor] = None,
                pad_mask: tp.Optional[torch.Tensor] = None) -> tp.Tuple[torch.Tensor, ...]:
        residual = audios
        audios = self.input_layernorm(audios)

        # self attn
        audios = self.self_attn(query=audios, key=audios, value=audios, pos_emb=pos_emb,
                                pad_mask=pad_mask)
        audios = residual + audios

        residual = audios
        audios = self.post_attention_layernorm(audios)

        # ffn
        audios = self.mlp(audios)

        audios = residual + audios
        return audios


class TransformerEncoder(torch.nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs: tp.Dict) -> None:
        super().__init__(**kwargs)
        self._config = config

        assert config.self_attention_model == "chunk_rotary"
        self.pos_enc = ChunkRotaryPositionalEmbedding(
            dim=config.d_model // config.n_heads,
            base=config.pos_emb_max_len,
            attn_chunk_size=config.attn_chunk_size,
            max_position_seq_len=config.max_position_seq_len,
        )

        # pre encode
        if config.subsampling == "stacking":
            self.pre_encode = StackingSubsampler(
                subsampling_factor=config.subsampling_factor,
                feat_in=config.feat_in,
                feat_out=config.d_model,
            )
        else:
            self.pre_encode = ConvSubsampling(
                subsampling_factor=config.subsampling_factor,
                feat_in=config.feat_in,
                feat_out=config.d_model,
                conv_channels=config.subsampling_conv_channels,
                subsampling=config.subsampling,
            )

        # layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerSpeech(config) for _ in range(config.num_hidden_layers)]
        )

        # norm
        norm_class = resolve_norm_class(config.norm_type)
        self.norm: nn.Module = norm_class(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.init_device
        )

    def set_max_audio_length(self,  seq_length: int,
                             device: tp.Optional[tp.Union[str, torch.device]]) -> None:
        device = "cpu" if device is None else device
        self.pos_enc.extend_pe(seq_length, device)

    def forward(self, audios: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        """Transformer based speech encoder.
        Args:
            audios (torch.Tensor): bs x seq_len x n_feat
            lengths (torch.Tensor): bs
        """
        self.set_max_audio_length(seq_length=audios.shape[1], device=audios.device)

        # subampling
        if hasattr(self, "pre_encode"):
            audios, lengths = self.pre_encode(audios, lengths)

        # pos embs
        audios, pos_emb = self.pos_enc(x=audios)

        # create pad mask
        max_audio_length = audios.size(1)
        pad_mask = torch.arange(0, max_audio_length, device=audios.device).expand(
            lengths.size(0), -1
        ) < lengths.unsqueeze(-1)
        pad_mask = ~pad_mask

        # layers fwd
        for layer in self.layers:
            audios = layer(audios, pos_emb=pos_emb, pad_mask=pad_mask)

        return audios, lengths
