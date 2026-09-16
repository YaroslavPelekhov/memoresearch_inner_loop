import warnings
import typing as tp

import torch

import deep_gemm


def _deep_gemm_version_check() -> None:
    major, minor, *_ = deep_gemm.__version__.split(".")
    assert (int(major), int(minor)) == (2, 3), (
        f"deep_gemm version must be 2.3.*, got {deep_gemm.__version__}"
    )

try:
    _deep_gemm_version_check()
except AssertionError:
    warnings.warn(f"for optimal performance, deep_gemm version must be 2.3.*, got {deep_gemm.__version__}")


class _deep_gemm_handle:
    def __init__(self, num_sms: tp.Optional[int] = None):
        p = torch.cuda.get_device_properties(0)
        assert (p.major, p.minor) == (9, 0)
        if num_sms is not None:
            deep_gemm.set_num_sms(int(num_sms))

    def k_grouped_fp8_gemm_contiguous(
        self,
        a: tp.Tuple[torch.Tensor, torch.Tensor],
        b: tp.Tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        ks: tp.List[int],
        ks_tensor: torch.Tensor
    ) -> None:
        assert (a[0].is_cuda and a[1].is_cuda and b[0].is_cuda and b[1].is_cuda and d.is_cuda and ks_tensor.is_cuda)
        assert (a[0].dtype == b[0].dtype == torch.float8_e4m3fn)
        # assert (a[1].dtype == b[1].dtype == d.dtype == torch.float32)
        assert (a[1].dtype == b[1].dtype == torch.float32)
        assert d.dtype in [torch.float32, torch.bfloat16]

        a_data, a_sf = a
        assert a_data.ndim == 1 and a_sf.ndim == 2

        b_data, b_sf = b
        assert b_data.ndim == 1 and b_sf.ndim == 2

        num_groups, _, _ = d.shape
        assert len(ks) == ks_tensor.shape[0] == num_groups
        assert a[0].is_contiguous() and b[0].is_contiguous()

        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, d)

    def m_grouped_fp8_gemm_nt_contiguous(
        self,
        a: tp.Tuple[torch.Tensor, torch.Tensor],
        b: tp.Tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        m_indices: torch.Tensor
    ) -> None:
        assert (a[0].is_cuda and a[1].is_cuda and b[0].is_cuda and b[1].is_cuda and d.is_cuda and m_indices.is_cuda)
        assert (a[0].dtype == b[0].dtype == torch.float8_e4m3fn), f"{a[0].dtype=} {b[0].dtype=}"
        assert (a[1].dtype == b[1].dtype == torch.float32)
        assert (d.dtype == torch.bfloat16)
        assert (a[0].is_contiguous() and b[0].is_contiguous() and d.is_contiguous())
        assert (a[0].shape[0] == m_indices.numel()), f"{a[0].shape[0]=} {m_indices.numel()=}"

        seq_len_a0, hid_dim_a0 = a[0].shape
        seq_len_a1, hid_dim_a1 = a[1].shape

        assert seq_len_a0 == seq_len_a1
        assert hid_dim_a0 // 128 == hid_dim_a1

        num_groups, int_dim, hid_dim = b[0].shape
        assert int_dim == d.shape[1]

        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)
