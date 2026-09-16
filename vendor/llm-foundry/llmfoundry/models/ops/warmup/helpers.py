import typing as tp

import torch

from llmfoundry.models.ops.warmup.utils import _BLOCK, _MAX_GROUPS_DEEPGEMM, _round_up


def _deepgemm_layout(
    total_tokens: int,
    device: torch.device,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Build a valid contiguous-group layout for row2col_deepgemm warmup.

    Returns:
      m_indices: int32 tensor of shape (total_tokens,)
          Group id for each token. Group ids are constant within each 128-token block.
      group_sizes: int32 tensor of shape (_MAX_GROUPS_DEEPGEMM,)
          Per-group sizes in TOKENS. Unused tail is zero-padded.
    """
    if total_tokens % _BLOCK != 0:
        raise ValueError(f"total_tokens must be divisible by {_BLOCK}, got {total_tokens}")

    num_blocks = total_tokens // _BLOCK
    num_groups = min(num_blocks, _MAX_GROUPS_DEEPGEMM)

    # Distribute blocks across groups as evenly as possible.
    base_blocks, rem_blocks = divmod(num_blocks, num_groups)

    group_blocks = torch.full(
        (_MAX_GROUPS_DEEPGEMM,),
        0,
        device=device,
        dtype=torch.int32,
    )
    group_blocks[:num_groups] = base_blocks
    if rem_blocks:
        group_blocks[:rem_blocks] += 1

    # Kernel expects per-group sizes in TOKENS, not in blocks.
    group_sizes = group_blocks * _BLOCK

    # Kernel reads m_indices at block starts: m_indices[pid_row * BLOCK_SIZE].
    # So we assign one group id per block, then expand to token-level length.
    block_group_ids = torch.repeat_interleave(
        torch.arange(num_groups, device=device, dtype=torch.int32),
        group_blocks[:num_groups],
    )
    assert block_group_ids.numel() == num_blocks

    m_indices = torch.repeat_interleave(block_group_ids, _BLOCK)
    assert m_indices.numel() == total_tokens
    assert int(group_sizes.sum().item()) == total_tokens

    return m_indices, group_sizes

def _build_permute_layout(
    num_tokens: int,
    num_experts: int,
    device: torch.device,
    *,
    n_routed: int = 1,
    tokens_per_expert: tp.Optional[tp.Sequence[int]] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor, int]:
    """Build routing tables for permute/unpermute warmup.

    Returns:
      row_id_map:    int32 tensor of shape (num_tokens, 2 * num_experts + 1)
      dst_to_padded: int32 tensor of shape (total_padded,)
      total_padded:  total padded rows across experts

    Layout contract expected by kernels:
      row_id_map[t, idx]                 -> dst_row for route idx
      row_id_map[t, num_experts + idx]   -> expert index for route idx
      row_id_map[t, 2 * num_experts]     -> n_routed for token t

      dst_to_padded[dst_row]             -> padded_dst_row

    Notes:
      - If tokens_per_expert is None, routes are distributed as evenly as possible.
      - If n_routed > 1, each token gets n_routed distinct experts.
      - total unpadded assignments = sum(tokens_per_expert) = num_tokens * n_routed.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if n_routed <= 0:
        raise ValueError(f"n_routed must be > 0, got {n_routed}")
    if n_routed > num_experts:
        raise ValueError(
            f"n_routed={n_routed} cannot exceed num_experts={num_experts}"
        )

    total_assignments = num_tokens * n_routed

    if tokens_per_expert is None:
        base, rem = divmod(total_assignments, num_experts)
        tokens_per_expert = [base + (1 if e < rem else 0) for e in range(num_experts)]
    else:
        tokens_per_expert = [int(x) for x in tokens_per_expert]
        if len(tokens_per_expert) != num_experts:
            raise ValueError(
                f"tokens_per_expert must have len={num_experts}, got {len(tokens_per_expert)}"
            )
        if any(x < 0 for x in tokens_per_expert):
            raise ValueError("tokens_per_expert must be non-negative")
        if sum(tokens_per_expert) != total_assignments:
            raise ValueError(
                "sum(tokens_per_expert) must equal num_tokens * n_routed, got "
                f"{sum(tokens_per_expert)} != {total_assignments}"
            )

    # Padded per-expert capacities, same alignment idea as runtime helper paths.
    tokens_per_expert_padded = [_round_up(t, _BLOCK) for t in tokens_per_expert]

    # Prefix sums in unpadded and padded expert layouts.
    cum: tp.List[int] = [0]
    cum_padded: tp.List[int] = [0]
    for t, t_pad in zip(tokens_per_expert, tokens_per_expert_padded):
        cum.append(cum[-1] + t)
        cum_padded.append(cum_padded[-1] + t_pad)
    total_padded = cum_padded[-1]

    # row_id_map columns:
    #   [0:num_experts)           -> dst_row slots
    #   [num_experts:2*num_experts) -> expert_idx slots
    #   [2*num_experts]           -> n_routed
    row_id_map = torch.zeros(
        (num_tokens, 2 * num_experts + 1),
        device=device,
        dtype=torch.int32,
    )

    # Assign routes token-by-token, keeping expert usage close to requested tokens_per_expert.
    remaining = list(tokens_per_expert)
    assigned_per_expert = [0] * num_experts
    cursor = 0

    for t in range(num_tokens):
        used_experts: set[int] = set()

        for route_idx in range(n_routed):
            chosen = None
            for _ in range(num_experts):
                e = cursor
                cursor = (cursor + 1) % num_experts
                if remaining[e] > 0 and e not in used_experts:
                    chosen = e
                    break

            if chosen is None:
                # This should only happen for invalid tokens_per_expert / n_routed combos.
                raise RuntimeError(
                    f"Could not assign route {route_idx} for token {t}; "
                    f"remaining={remaining}"
                )

            dst_row = cum[chosen] + assigned_per_expert[chosen]

            row_id_map[t, route_idx] = dst_row
            row_id_map[t, num_experts + route_idx] = chosen
            used_experts.add(chosen)

            assigned_per_expert[chosen] += 1
            remaining[chosen] -= 1

        row_id_map[t, 2 * num_experts] = n_routed

    if any(x != 0 for x in remaining):
        raise RuntimeError(f"Unassigned expert slots remain: {remaining}")

    # Map unpadded dst_row -> padded dst_row.
    # Kernel indexes this by dst_row/src_row coming from row_id_map.
    dst_to_padded = torch.zeros(total_padded, device=device, dtype=torch.int32)
    for e in range(num_experts):
        for local_idx in range(tokens_per_expert[e]):
            dst_to_padded[cum[e] + local_idx] = cum_padded[e] + local_idx

    return row_id_map, dst_to_padded, total_padded
