import torch
import torch.nn as nn

from typing import Optional, Any

from .base_projector import BaseProjector


class PixelShuffleProjector(BaseProjector):
    def __init__(
        self,
        mm_hidden_size: int = 1024,
        hidden_size: int = 4096,
        **kwargs: Any,
    ):
        """
        Pixel-shuffle projector.

        Args:
            mm_hidden_size: Hidden size of the input tensor.
            hidden_size: Hidden size of the output tensor.
        Kwargs:
            downsample_ratio: Downsample ratio.
            use_reduction: Whether to use reduction. (Required for video)
            avg_tokens_per_frame: Average number of tokens per frame. (Required if use_reduction is True)
            video_token: Token for the video. (Required if use_reduction is True)
            num_frames: Number of frames per video (same as num_frames in the video processor). (Required if use_reduction is True)
            frame_aspect_ratio: Aspect ratio of the frame. (Required if use_reduction is True)
        """
        super().__init__(mm_hidden_size, hidden_size)

        self.downsample_ratio = kwargs.pop("downsample_ratio", 0.5)
        self.use_reduction = kwargs.pop("use_reduction", False)
        if self.use_reduction:
            self.avg_tokens_per_frame = kwargs.pop("avg_tokens_per_frame")
            self.video_token = kwargs.pop("video_token")
            self.num_frames = kwargs.pop("num_frames")
            self.frame_aspect_ratio = kwargs.pop("frame_aspect_ratio")

        # Final MLP
        self.module_f = nn.Sequential(
            nn.LayerNorm(mm_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                mm_hidden_size * int(1 / self.downsample_ratio) ** 2, hidden_size
            ),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        N, W, H, C = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(
            N,
            W,
            int(H * self.downsample_ratio),
            int(C / self.downsample_ratio),
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.reshape(
            N,
            int(H * self.downsample_ratio),
            int(W * self.downsample_ratio),
            int(C / (self.downsample_ratio**2)),
        )
        # N, H * scale, W * scale, C // (scale ** 2) -> N, W * scale, H * scale, C // (scale ** 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x  # N, W * scale, H * scale, C // (scale ** 2)

    def _get_patchwise_compression_mask(self, videos, num_to_keep):
        """
        Args:
            videos: torch.Tensor of shape (B, T, P, D)
            num_to_keep: Number of patches to keep.

        Returns:
            bool mask (B, T, P).
        """
        B, T, P, D = videos.shape
        diffs = (
            (videos[:, 1:, :, :] - videos[:, :-1, :, :]).abs().mean(dim=-1)
        )  # (B, T-1, P)
        first = torch.zeros((B, 1, P), device=videos.device)
        diffs = torch.cat([first, diffs], dim=1)

        flat_batched = diffs.view(B, -1)

        # select top k patches
        _, topk_indices = torch.topk(flat_batched, k=num_to_keep, dim=-1)

        # masking
        mask = torch.zeros_like(flat_batched, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        mask = mask.view(B, T, P)

        # save first frame:
        mask[:, 0, :] = True
        return mask

    def _compress_video_tokens(
        self,
        compression_mask: torch.BoolTensor,  # (T, P)
        video_embeds: torch.FloatTensor,  # (T, P, D)
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        video_embeds = video_embeds[compression_mask]

        video_selected = input_ids == self.video_token
        text_mask = torch.logical_not(video_selected)

        text_mask[video_selected] = compression_mask.flatten()[: video_selected.sum()]

        input_ids = input_ids[text_mask]

        if attention_mask is not None:
            attention_mask = attention_mask[text_mask]
        if labels is not None:
            labels = labels[text_mask]
            labels[input_ids == self.video_token] = -100

        return video_embeds, input_ids, attention_mask, labels

    def apply_compression_video_token(
        self,
        input_ids: torch.Tensor,
        video_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        assert self.frame_aspect_ratio == "pad", (
            "Token reduction doesn't support several crops for frame!"
        )
        BT, P, D = video_embeds.shape

        # dummy video case, just return
        if BT == 1:
            return video_embeds, input_ids, attention_mask, labels

        assert input_ids.shape[0] == 1, (
            f"Token reduction works only with batch_size = 1, got {input_ids.shape[0]=}"
        )

        B, T = 1, BT
        num_to_keep = self.avg_tokens_per_frame * T

        video_embeds = video_embeds.view(B, T, P, -1)
        compression_mask = self._get_patchwise_compression_mask(
            video_embeds, num_to_keep
        )

        video_embeds, input_ids, attention_mask, labels = self._compress_video_tokens(
            compression_mask, video_embeds, input_ids, attention_mask, labels
        )

        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
        return video_embeds, input_ids, attention_mask, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pixel shuffle
        H = W = int(x.shape[1] ** 0.5)  # H = W = int(seq_len ** 0.5)
        # 1. (N, seq_len, mm_hidden_size) -> (N, seq_len**0.5, seq_len**0.5, mm_hidden_size)
        x = x.reshape(x.shape[0], H, W, -1)
        # 2. (N, seq_len**0.5, seq_len**0.5, mm_hidden_size) -> (N, seq_len**0.5 * scale, seq_len**0.5 * scale, mm_hidden_size // scale**2)
        x = self.pixel_shuffle(x)
        # 3. (N, seq_len**0.5 * scale, seq_len**0.5 * scale, mm_hidden_size // scale**2) -> (N, seq_len * scale**2, mm_hidden_size // scale**2)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # MLP
        # 4. (N, seq_len * scale**2, mm_hidden_size // scale**2) -> (N, seq_len * scale**2, hidden_size)
        return self.module_f(x)
