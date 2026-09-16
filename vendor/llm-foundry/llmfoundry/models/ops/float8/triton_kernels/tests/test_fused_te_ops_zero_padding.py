"""Correctness tests for the torch.empty + _zero_expert_padding_kernel
optimisation in fused_te_ops.py.

Run with:
    cd llmfoundry/models/ops/float8/triton_kernels/tests
    pytest test_fused_te_ops_zero_padding.py -v
"""
import pytest
import torch

from utils import _te_version_check, _setup_standalone_path

_te_version_check()
_setup_standalone_path()

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from llmfoundry.models.ops.float8.triton_kernels.fused_te_ops import (
    _launch_zero_expert_padding,
    permute_and_pad_fn,
    unpad_and_unpermute_fn,
)

DEVICE = torch.device("cuda")
ALIGN_SIZE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pad_info(tokens_per_expert, align_size=ALIGN_SIZE):
    """Return (pad_starts_t, pad_counts_t) as CUDA int32 tensors."""
    padded_offset = 0
    pad_starts, pad_counts = [], []
    for t in tokens_per_expert:
        p = (t + align_size - 1) // align_size * align_size
        pad_starts.append(padded_offset + t)
        pad_counts.append(p - t)
        padded_offset += p
    pad_starts_t = torch.tensor(pad_starts, device=DEVICE, dtype=torch.int32)
    pad_counts_t = torch.tensor(pad_counts, device=DEVICE, dtype=torch.int32)
    return pad_starts_t, pad_counts_t


def _total_tokens_padded(tokens_per_expert, align_size=ALIGN_SIZE):
    return sum(
        (t + align_size - 1) // align_size * align_size
        for t in tokens_per_expert
    )


def _make_local_map(tokens_per_expert, num_experts, device=DEVICE):
    """Build a boolean routing map (num_tokens x num_experts)."""
    rows = []
    for expert_id, count in enumerate(tokens_per_expert):
        for _ in range(count):
            row = torch.zeros(num_experts, dtype=torch.bool, device=device)
            row[expert_id] = True
            rows.append(row)
    if not rows:
        return torch.zeros((0, num_experts), dtype=torch.bool, device=device)
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Unit tests: _zero_expert_padding_kernel via _launch_zero_expert_padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens_per_expert,hidden_size", [
    ([1] * 4,         256),   # worst-case: 127/128 rows are padding
    ([128] * 4,       512),   # no padding needed (pad_count == 0)
    ([0, 64, 128, 1], 256),   # expert with 0 tokens
    ([500, 300, 1],   1024),  # realistic sizes
])
def test_zero_expert_padding_kernel_2d(tokens_per_expert, hidden_size):
    """Padding rows must be zeroed; real-token rows must remain unchanged."""
    total = _total_tokens_padded(tokens_per_expert)
    if total == 0:
        pytest.skip("empty tensor")

    SENTINEL = 7.0
    buf = torch.full((total, hidden_size), SENTINEL, device=DEVICE, dtype=torch.float32)
    pad_starts_t, pad_counts_t = _make_pad_info(tokens_per_expert)

    _launch_zero_expert_padding(buf, pad_starts_t, pad_counts_t)

    padded_offset = 0
    for t, p_count in zip(tokens_per_expert, pad_counts_t.tolist()):
        p = (t + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        pad_start = padded_offset + t
        if p_count > 0:
            assert buf[pad_start : pad_start + p_count].eq(0).all(), \
                f"Padding rows [{pad_start}:{pad_start+p_count}] not zeroed"
        if t > 0:
            assert buf[padded_offset : padded_offset + t].eq(SENTINEL).all(), \
                f"Real-token rows [{padded_offset}:{padded_offset+t}] were corrupted"
        padded_offset += p


@pytest.mark.parametrize("tokens_per_expert", [
    [1] * 4, [128] * 3, [0, 64, 1], [500, 1],
])
def test_zero_expert_padding_kernel_1d(tokens_per_expert):
    """1-D tensor (permuted_probs shape): padding positions must be zeroed."""
    total = _total_tokens_padded(tokens_per_expert)
    if total == 0:
        pytest.skip("empty tensor")

    SENTINEL = 3.0
    buf = torch.full((total,), SENTINEL, device=DEVICE, dtype=torch.float32)
    pad_starts_t, pad_counts_t = _make_pad_info(tokens_per_expert)

    _launch_zero_expert_padding(buf, pad_starts_t, pad_counts_t)

    padded_offset = 0
    for t, p_count in zip(tokens_per_expert, pad_counts_t.tolist()):
        p = (t + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        pad_start = padded_offset + t
        if p_count > 0:
            assert buf[pad_start : pad_start + p_count].eq(0).all(), \
                f"1-D: Padding positions [{pad_start}:{pad_start+p_count}] not zeroed"
        if t > 0:
            assert buf[padded_offset : padded_offset + t].eq(SENTINEL).all(), \
                f"1-D: Real positions [{padded_offset}:{padded_offset+t}] were corrupted"
        padded_offset += p


def test_zero_expert_padding_kernel_all_zero_counts():
    """When every expert exactly fills its block, no stores happen and the
    buffer (filled with SENTINEL) must remain fully unchanged."""
    tokens_per_expert = [128, 256, 128]
    total = _total_tokens_padded(tokens_per_expert)
    SENTINEL = 5.0
    buf = torch.full((total, 64), SENTINEL, device=DEVICE, dtype=torch.float32)
    pad_starts_t, pad_counts_t = _make_pad_info(tokens_per_expert)
    assert pad_counts_t.sum().item() == 0

    _launch_zero_expert_padding(buf, pad_starts_t, pad_counts_t)
    assert buf.eq(SENTINEL).all(), "Buffer should be untouched when all pad_counts are 0"


# ---------------------------------------------------------------------------
# End-to-end: permute_and_pad_fn — padding rows are zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens_per_expert,hidden_size", [
    ([1, 1, 1, 1],    256),
    ([128] * 4,       512),
    ([0, 64, 128, 1], 256),
    ([500, 0, 300, 1], 1024),
    ([127, 129, 64],  256),
])
def test_permute_and_pad_padding_rows_are_zero(tokens_per_expert, hidden_size):
    """output and permuted_probs must have zeros in all padding rows."""
    num_experts = len(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    if total_tokens == 0:
        pytest.skip("no tokens")

    local_map = _make_local_map(tokens_per_expert, num_experts)
    recv_x = torch.randn(total_tokens, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, num_experts, device=DEVICE, dtype=torch.float32)
    probs = probs * local_map.float()
    tokens_per_expert_t = torch.tensor(tokens_per_expert, device=DEVICE, dtype=torch.int32)

    output, permuted_probs, _, _, pad_starts_t, pad_counts_t = \
        permute_and_pad_fn(recv_x, local_map, probs, tokens_per_expert_t)

    padded_offset = 0
    for t, p_count in zip(tokens_per_expert, pad_counts_t.tolist()):
        p = (t + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        pad_start = padded_offset + t
        if p_count > 0:
            assert output[pad_start : pad_start + p_count].eq(0).all(), \
                "permute output: padding rows not zero"
            assert permuted_probs[pad_start : pad_start + p_count].eq(0).all(), \
                "permuted_probs: padding positions not zero"
        padded_offset += p


# ---------------------------------------------------------------------------
# End-to-end: round-trip permute → unpermute recovers original tensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens_per_expert,hidden_size", [
    ([4, 4, 4, 4],   256),
    ([128] * 2,      512),
    ([0, 64, 1],     128),
    ([500, 300, 1],  256),
])
def test_round_trip_permute_unpermute(tokens_per_expert, hidden_size):
    """unpad_and_unpermute_fn(permute_and_pad_fn(x)) must recover x
    for exclusive single-expert routing (n_routed == 1 per token)."""
    num_experts = len(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    if total_tokens == 0:
        pytest.skip("no tokens")

    local_map = _make_local_map(tokens_per_expert, num_experts)
    recv_x = torch.randn(total_tokens, hidden_size, device=DEVICE, dtype=torch.bfloat16)
    probs = torch.rand(total_tokens, num_experts, device=DEVICE, dtype=torch.float32)
    probs = probs * local_map.float()
    tokens_per_expert_t = torch.tensor(tokens_per_expert, device=DEVICE, dtype=torch.int32)

    permuted, _, row_id_map, row_to_padded_row_id, pad_starts_t, pad_counts_t = \
        permute_and_pad_fn(recv_x, local_map, probs, tokens_per_expert_t)

    recovered = unpad_and_unpermute_fn(
        permuted.to(torch.bfloat16),
        row_id_map,
        row_to_padded_row_id,
        recv_x.shape,
        pad_starts_t,
        pad_counts_t,
    )

    assert recovered.shape == recv_x.shape
    torch.testing.assert_close(
        recovered.float(), recv_x.float(),
        atol=1e-2, rtol=1e-2,
        msg="Round-trip permute→unpermute did not recover original tensor",
    )


# ---------------------------------------------------------------------------
# Backward: grad_input must have zeros in padding rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tokens_per_expert,hidden_size", [
    ([4, 4, 4, 4], 128),
    ([1] * 4,      128),
    ([0, 64, 1],   128),
])
def test_unpad_unpermute_backward_padding_rows_zero(tokens_per_expert, hidden_size):
    """The gradient flowing back through unpad_and_unpermute_fn into the padded
    space must be zero in all alignment-padding rows."""
    num_experts = len(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    if total_tokens == 0:
        pytest.skip("no tokens")

    local_map = _make_local_map(tokens_per_expert, num_experts)
    recv_x = torch.randn(total_tokens, hidden_size, device=DEVICE, dtype=torch.float32)
    probs = torch.rand(total_tokens, num_experts, device=DEVICE, dtype=torch.float32)
    probs = probs * local_map.float()
    tokens_per_expert_t = torch.tensor(tokens_per_expert, device=DEVICE, dtype=torch.int32)

    permuted, _, row_id_map, row_to_padded_row_id, pad_starts_t, pad_counts_t = \
        permute_and_pad_fn(recv_x, local_map, probs, tokens_per_expert_t)

    permuted_f = permuted.float().detach().requires_grad_(True)
    recovered = unpad_and_unpermute_fn(
        permuted_f, row_id_map, row_to_padded_row_id,
        recv_x.shape, pad_starts_t, pad_counts_t,
    )
    recovered.sum().backward()

    grad = permuted_f.grad  # (total_tokens_padded, hidden_size)
    padded_offset = 0
    for t, p_count in zip(tokens_per_expert, pad_counts_t.tolist()):
        p = (t + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE
        pad_start = padded_offset + t
        if p_count > 0:
            assert grad[pad_start : pad_start + p_count].eq(0).all(), \
                "Backward: padding rows of grad_input are not zero"
        padded_offset += p
