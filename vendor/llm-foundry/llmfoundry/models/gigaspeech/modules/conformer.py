import logging
import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import PretrainedConfig
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.layers.rotary import apply_rotary_emb
from einops import rearrange
from itertools import chain


class ConformerConfig(PretrainedConfig):
    model_type: str = "conformer"
    DEFAULT_CHUNK_SIZE: int = 200
    ALLOWED_ATTN_TYPES: tp.List[str] = ["chunk_rotary"]
    ALLOWED_SUBSAMPLING_TYPES: tp.List[str] = ["striding_conv1d", "striding", "stacking"]

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 24,
        d_model: int = 1024,
        feat_out: int = -1,
        subsampling: str = "striding_conv1d",
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = -1,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "chunk_rotary",
        n_heads: int = 16,
        att_context_size: tp.Optional[tp.Union[int, tp.List[int]]] = None,
        att_context_style: str = "regular",
        pos_emb_max_len: int = 5000, # todo fedorovgv выпилить
        conv_kernel_size: int = 31,
        conv_norm_type: str = "layer_norm",
        dropout: float = 0.1,
        dropout_emb: float = 0.1,
        dropout_att: float = 0.1,
        attn_chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_position_seq_len: int = 131072,
        use_causal_conv_chunking: bool = False,
        use_causal_conv: bool = False,
        window_size: tp.Tuple[int, int] = (-1, -1),
        **kwargs: tp.Dict,
    ) -> None:
        """Config class.
        """
        super().__init__(**kwargs)

        self.feat_in = feat_in
        self.n_layers = n_layers
        self.d_model = d_model
        self.feat_out = feat_out

        assert subsampling in self.ALLOWED_SUBSAMPLING_TYPES, (
            f"Only {self.ALLOWED_SUBSAMPLING_TYPES} allowed, but found {subsampling} !"
        )
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels

        self.ff_expansion_factor = ff_expansion_factor

        assert self_attention_model in self.ALLOWED_ATTN_TYPES, (
            f"Only {self.ALLOWED_ATTN_TYPES} allowed, but found {self_attention_model} !"
        )
        self.self_attention_model = self_attention_model
        self.attn_chunk_size = attn_chunk_size
        self.n_heads = n_heads
        self.att_context_size = att_context_size
        self.att_context_style = att_context_style

        self.conv_kernel_size = conv_kernel_size
        self.conv_norm_type = conv_norm_type

        self.dropout = dropout
        self.dropout_emb = dropout_emb
        self.dropout_att = dropout_att

        self.use_causal_conv_chunking = use_causal_conv_chunking
        self.use_causal_conv = use_causal_conv
        self.window_size = tuple(window_size)
        self.pos_emb_max_len = pos_emb_max_len
        self.max_position_seq_len = max_position_seq_len


class StackingSubsampler(nn.Module):
    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int) -> None:
        super().__init__()

        self.feat_in = feat_in
        self.feat_out = feat_out
        self.subsampling_factor = subsampling_factor

        self.proj = nn.Linear(in_features=self.subsampling_factor * feat_in, out_features=feat_out,
                              bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.tensor, ...]:
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py#L25
        b, t, h = x.size()

        mask = torch.arange(t, device=x.device).unsqueeze(0).unsqueeze(-1).expand(b, t, h)
        mask = mask >= lengths.view(b, 1, 1)
        x = x.clone()
        x[mask] = 0

        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        lengths = torch.div(lengths + pad_size + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode='floor')

        x = self.proj(x)

        return x, lengths


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
        subsampling: tp.Optional[str] = None,
    ) -> None:
        super().__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._subsampling_factor = subsampling_factor
        self._sampling_num = int(math.log(subsampling_factor, 2))

        self.activation = nn.ReLU(inplace=True)

        assert subsampling in ["striding", "striding_conv1d"]
        self.subsampling = subsampling

        layers = []
        if subsampling == 'striding':
            in_channels = 1

            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=(self._left_padding, self._right_padding),
                    )
                )
                layers.append(self.activation)
                in_channels = conv_channels

            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True

        elif subsampling == 'striding_conv1d':
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2
            self._max_cache_len = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
                layers.append(self.activation)
                in_channels = conv_channels

            self.conv2d_subsampling = False

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: tp.Optional[torch.Tensor] = None):
        if lengths is None:
            lengths = torch.ones(x.size(0), device=x.device).long() * x.size(1)

        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
            x = self.conv(x)
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        else:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)

        return x, lengths


class ConformerFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float,
            activation: str = 'silu',
        ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        if activation == 'silu':
            activation_md = nn.SiLU(inplace=True)
        else:
            raise ValueError('Only silu activation supported!')

        self.activation = activation_md
        # self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x


class NemoConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: tp.Union[str, int] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        if isinstance(padding, int):
            self._left_padding = padding
            self._right_padding = padding
        elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
            self._left_padding = padding[0]
            self._right_padding = padding[1]
        else:
            raise ValueError(f"Invalid padding param: {padding}!")

        self._left_cache_len = padding
        self._right_cache_len = padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor):
        base_dtype = x.dtype
        x = nn.functional.pad(x, pad=(self._left_padding, self._right_padding))
        x = x.to(base_dtype)
        return super().forward(x)


class ConformerConvolution(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        conv_context_size: tp.Union[int, tp.List[int]],
        norm_type: str = "batch_norm",
        causal_conv_chunk: tp.Optional[int] = None,
        use_causal_conv: bool = False,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        assert not (use_causal_conv and causal_conv_chunk), \
            "Cannot use causal conv with causal conv chunking"

        if isinstance(conv_context_size, int):
            conv_context_size = [conv_context_size, conv_context_size]

        if causal_conv_chunk is not None:
            super_conv_context = 0
        elif use_causal_conv:
            super_conv_context = [kernel_size - 1, 0]
        else:
            super_conv_context = conv_context_size

        assert conv_context_size is not None, f'conv_context_size is None'
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.depthwise_conv = NemoConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=super_conv_context,
            groups=d_model,
            bias=True,
        )

        assert norm_type in ["batch_norm", "layer_norm"]
        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.batch_norm = nn.LayerNorm(d_model)

        self.activation = nn.SiLU(inplace=True)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.causal_conv_chunk = causal_conv_chunk
        self.conv_ctx_size = conv_context_size[0]
        if self.causal_conv_chunk is not None:
            assert conv_context_size[0] == conv_context_size[1], \
                f"Expected sym values in paddings {conv_context_size} for chunking"

    def forward(self, x: torch.Tensor, pad_mask: tp.Optional[torch.Tensor] = None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        if self.causal_conv_chunk is not None:
            b, c, t = x.shape
            chunk_size = self.causal_conv_chunk
            assert chunk_size >= self.conv_ctx_size, \
                f'Expected chunk {chunk_size} >= conv context {self.conv_ctx_size}'
            x = F.pad(x, pad=(0, (chunk_size - t % chunk_size) % chunk_size))  # t_new % chunk_size == 0
            assert x.shape[2] % chunk_size == 0
            n_chunks = x.shape[2] // chunk_size
            x = x.reshape(b, c, n_chunks, chunk_size)
            cached = x[:, :, :, -self.conv_ctx_size:].roll(1, dims=-2) # [b, c, N, ctx_size]
            cached[:, :, 0, :] = 0
            x = F.pad(torch.cat([cached, x], dim=-1), (0, self.conv_ctx_size)) # [b, c, N, ctx_size + chunk_size + ctx_size]
            x = x.transpose(1, 2).reshape(b * n_chunks, c, -1)
            x = self.depthwise_conv(x)
            x = x.reshape(b, n_chunks, c, -1).transpose(1, 2).reshape(b, c, n_chunks * chunk_size)
            x = x[:, :, :t]
            assert x.shape[-1] == t, f'{x.shape}[-1] != {b, c, t}[-1]'
        else:
            x = self.depthwise_conv(x)

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ChunkRotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim: int, base: int, attn_chunk_size: int, max_position_seq_len: int) -> None:
        """Chunk-wise rotary positional embedding
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            attn_chunk_size: Size of the chunk
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.attn_chunk_size = attn_chunk_size
        self.max_position_seq_len = max_position_seq_len

        self.coss, self.sins = self.create_pe(self.max_position_seq_len, "cpu")

    def create_pe(self, length: int, device: tp.Union[str, torch.device]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Create sin, cos chunks pe tables of size chunk_size x dim//2.
        """
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(self.attn_chunk_size, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        blocks = math.ceil(length / self.attn_chunk_size)
        coss = torch.cat([cos for _ in range(blocks)], dim=0).to(device)
        sins = torch.cat([sin for _ in range(blocks)], dim=0).to(device)
        return coss, sins

    def extend_pe(self, length: int, device: tp.Union[str, torch.device]) -> None:
        if hasattr(self, "coss") and self.coss.shape[0] >= length:
            self.coss, self.sins = self.coss.to(device), self.sins.to(device)
            return None
        self.coss, self.sins = self.create_pe(length, device)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, ...]]:
        seq_len = x.shape[1]
        coss = self.coss[:seq_len, :]
        sins = self.sins[:seq_len, :]
        return x, (coss, sins)


class ChunkRotaryPositionMultiHeadedAttention(nn.Module):

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout: tp.Optional[float] = None,
        attn_chunk_size: tp.Optional[int] = None,
        window_size: tp.Tuple[int, int] = (-1, -1),
    ):
        """Construct a ChunkRotaryPositionMultiHeadedAttention object."""
        super().__init__()

        assert n_feat % n_head == 0

        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head

        self.attn_chunk_size = attn_chunk_size
        self.window_size = window_size
        if self.window_size != (-1, -1):
            logging.warning(f"Using non-default window_size = {self.window_size}")
        self.dropout_value = dropout if dropout else 0.0

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        **kwargs: tp.Dict,
    ) -> torch.Tensor:
        """Compute chunk-wise rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        B, T, _ = value.size()
        query = query.view(B, T, self.h, self.d_k)
        key = key.view(B, T, self.h, self.d_k)
        value = value.view(B, T, self.h, self.d_k)

        # query, key: seq_len x bs x nh x hd
        # cos, sin: seq_len x hd/2
        cos, sin = pos_emb
        cos, sin = cos.to(query.dtype), sin.to(query.dtype)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        query = query.view(B, T, self.h * self.d_k)
        key = key.view(B, T, self.h * self.d_k)
        value = value.view(B, T, self.h * self.d_k)

        n_batch, seqlen, _ = query.shape
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        dropout = self.dropout_value if self.training else 0.0

        pad_mask = ~pad_mask

        q_unpad, indices_q, _, max_seqlen_q, _ = unpad_input(q, pad_mask)
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=self.h)
        k_unpad, _, _, _, _ = unpad_input(k, pad_mask)
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=self.h)
        v_unpad, _, _, _, _ = unpad_input(v, pad_mask)
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=self.h)

        lenghts = pad_mask.sum(1)
        lengths_q = [
            [self.attn_chunk_size] * (t // self.attn_chunk_size) + ([t % self.attn_chunk_size]
            if t % self.attn_chunk_size != 0 else [])
            for t in lenghts.cpu().numpy()
        ]
        lengths_q = torch.tensor(list(chain.from_iterable(lengths_q)), dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.nn.functional.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)

        assert sum(lengths_q) == q_unpad.size(0), \
            f'sum of chunks lengths differ from total frames length {sum(lengths_q)} != {q_unpad.size(0)}'

        output_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_q,
            dropout_p=dropout,
            window_size=self.window_size,
        )
        out = pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, n_batch, seqlen)

        out = self.linear_out(out)
        return out


class ConformerLayer(nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str = "rotary",
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        conv_norm_type: str = "batch_norm",
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        attn_chunk_size: tp.Optional[tp.Any] = None,
        use_causal_conv_chunking: bool = False,
        use_causal_conv: bool = False,
        window_size: tp.Tuple[int, int] = (-1, -1),
        *args: tp.List,
        **kwargs: tp.Dict,
    ):
        super().__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=(conv_kernel_size - 1) // 2,
            causal_conv_chunk=attn_chunk_size if use_causal_conv_chunking else None,
            use_causal_conv=use_causal_conv,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)

        assert self_attention_model == "chunk_rotary"
        self.self_attn = ChunkRotaryPositionMultiHeadedAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout=dropout_att,
            attn_chunk_size=attn_chunk_size,
            window_size=window_size,
        )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        # self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: tp.Optional[torch.Tensor] = None,
        pos_emb: tp.Optional[torch.Tensor] = None,
        pad_mask: tp.Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.Tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        # residual = residual + self.dropout(x) * self.fc_factor
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)

        assert self.self_attention_model == "chunk_rotary"
        assert pos_emb is not None and att_mask is None and pad_mask is not None
        x = self.self_attn(query=x, key=x, value=x, pos_emb=pos_emb, pad_mask=pad_mask)

        # residual = residual + self.dropout(x)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)

        # residual = residual + self.dropout(x)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        # residual = residual + self.dropout(x) * self.fc_factor
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):

    def __init__(self, config: ConformerConfig, **kwargs: tp.Dict):
        super().__init__(**kwargs)

        if isinstance(config, dict):
            config = ConformerConfig(**config)

        d_ff = config.d_model * config.ff_expansion_factor
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self._feat_in = config.feat_in
        self.scale = math.sqrt(self.d_model)
        self.att_context_style = config.att_context_style
        self.subsampling_factor = config.subsampling_factor

        if config.att_context_size:
            self.att_context_size = list(config.att_context_size)
        else:
            self.att_context_size = [-1, -1]

        assert config.att_context_style == "regular"
        self.attn_chunk_size = None

        self.self_attention_model = config.self_attention_model

        assert self.self_attention_model == "chunk_rotary"
        self.pos_enc = ChunkRotaryPositionalEmbedding(
            dim=config.d_model // config.n_heads,
            base=config.pos_emb_max_len,
            attn_chunk_size=config.attn_chunk_size,
            max_position_seq_len=config.max_position_seq_len,
        )

        subsampling_conv_channels = config.subsampling_conv_channels

        if config.subsampling_conv_channels == -1:
            subsampling_conv_channels = config.d_model

        if config.subsampling and config.subsampling_factor > 1:
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
                    conv_channels=subsampling_conv_channels,
                    subsampling=config.subsampling,
                )

        self._feat_out = config.d_model

        self.pos_emb_max_len = config.pos_emb_max_len

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            layer = ConformerLayer(
                d_model=config.d_model,
                d_ff=d_ff,
                self_attention_model=config.self_attention_model,
                n_heads=config.n_heads,
                conv_kernel_size=config.conv_kernel_size,
                conv_norm_type=config.conv_norm_type,
                dropout=config.dropout,
                dropout_att=config.dropout_att,
                attn_chunk_size=config.attn_chunk_size,
                layer_index=i,
                use_causal_conv_chunking=config.use_causal_conv_chunking,
                use_causal_conv=config.use_causal_conv,
                window_size=config.window_size,
            )
            self.layers.append(layer)

        if config.feat_out > 0 and config.feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, config.feat_out)
            self._feat_out = config.feat_out
        else:
            self.out_proj = None
            self._feat_out = config.d_model

        self.set_max_audio_length(self.pos_emb_max_len, 'cpu')

    def set_max_audio_length(self, seq_length: int, device: tp.Optional[tp.Union[str, torch.device]]) -> None:
        """Sets maximum input length. Pre-calculates internal seq_range mask.
        """
        device = "cpu" if device is None else device
        self.pos_enc.extend_pe(seq_length, device)

    def forward(self, audio_signal: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        """Conformer encoder forward.

        Args:
            audio_signal (torch.Tensor): bs x seq_len x n_mel
            length (torch.Tensor): bs

        Returns:
            tp.Tuple[torch.Tensor, ...]: processed audio_signal and lengths.
        """
        max_audio_length: int = audio_signal.size(1)
        self.set_max_audio_length(seq_length=max_audio_length, device=audio_signal.device)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, lengths = self.pre_encode(x=audio_signal, lengths=lengths)

        max_audio_length = audio_signal.size(1)

        # Create the self-attention and padding masks
        if self.self_attention_model == "chunk_rotary":
            padding_length = lengths
            audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        else:
            raise ValueError("Wrong attention type.")

        # NOTE (Sbr, fedorovgv) : double negatiation (here and later) is necessary, it is the torch bug!
        pad_mask = torch.arange(0, max_audio_length, device=audio_signal.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)
        pad_mask = ~pad_mask

        for _, layer in enumerate(self.layers):
            audio_signal = layer(audio_signal, att_mask=None, pos_emb=pos_emb, pad_mask=pad_mask)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        return audio_signal, lengths


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):  # type: ignore
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
