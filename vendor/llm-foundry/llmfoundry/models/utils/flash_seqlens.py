import torch
import composer.utils.dist as dist
from llmfoundry.models.parallel.sequence.ring_flash_attn import llama3_flash_attn_prepare_cu_seqlens


def get_cu_seqlens_from_pos_ids(position_ids: torch.Tensor):
    """generate a cumulative sequence length mask for flash attention using pos ids"""
    # position_ids = position_ids.view(-1, seq_length).long()
    position_ids = position_ids.flatten()
    device = position_ids.device

    # Count the number of consecutive zeros from the right side
    padding_length = (position_ids == 0).int().flip(dims=[0]).cumprod(dim=0).sum().item()

    # Adjust the row to exclude padding
    adjusted_row = position_ids[:-padding_length] if padding_length else position_ids.clone()

    # Find where the position resets to 0 (indicating a new sequence)
    seq_starts = torch.cat(
        [
            torch.tensor([True], dtype=torch.bool, device=device),
            adjusted_row[1:] == 0,
        ]
    )
    # Get the indices where the sequence starts
    start_indices = torch.cat(
        [
            (seq_starts).nonzero(as_tuple=True)[0],
            torch.tensor([len(adjusted_row)], dtype=torch.int32, device=device),
        ]
    )
    # Calculate the sequence lengths
    seq_lengths = start_indices[1:] - start_indices[:-1]
    # Calculate the cumulative sequence lengths
    cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)])
    # Append the padding length to the cumulative sequence lengths
    if padding_length:
        cu_seqlens = torch.cat([cu_seqlens, torch.tensor([len(position_ids)], dtype=torch.int32, device=device)])
    max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().to(dtype=torch.int32)

    cu_seqlens = cu_seqlens.squeeze().to(dtype=torch.int32)
    # строчка ниже оч важна
    # flash-attention ожидает в параметре max_seq_len число
    # Мы раньше делали тензор с 1 элементом (который еще и на gpu лежит)
    # -> каждый раз при использовании переменной max_seq_len нужно было синкануться с gpu
    # -> блокировка исполнения и невозможность подкачки весов заранее
    max_seq_len = max_seq_len.detach().cpu().item()

    return cu_seqlens, max_seq_len


def get_llama3_cu_seqlens_from_pos_ids(position_ids: torch.Tensor):
    # cu_seqlens logiс from: 
    # https://github.com/huggingface/nanotron/blob/7bc9923285a03069ebffe994379a311aceaea546/src/nanotron/models/qwen.py#L767
    # start_indices = torch.where(position_ids.view(-1) == 0)[0]
    # cu_seqlens = torch.cat(
    #     [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
    # ).to(torch.int32)

    # our cu_seqlens
    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)

    (
        cu_seqlens_q, 
        cu_seqlens_k, 
        max_seqlen_q, 
        max_seqlen_k, 
        local_k_slice
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens, 
        causal=True, 
        rank=dist.get_sp_group_rank(), # type: ignore
        world_size=dist.get_sp_group_size() # type: ignore
    )

    cu_seqlens = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "local_k_slice": local_k_slice
    }

    max_seqlen = {
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k
    }

    return cu_seqlens, max_seqlen
