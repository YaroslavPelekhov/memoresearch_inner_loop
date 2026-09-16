"""CUDA capability check for SonicMoE (sm90+)."""

import torch


def require_sonic_moe_cuda_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SonicMoE requires CUDA.")

    d = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(d)
    if (major, minor) < (9, 0):
        raise RuntimeError(
            f"SonicMoE needs Hopper (sm90+); device {d} ({torch.cuda.get_device_name(d)}) "
            f"is capability {major}.{minor}."
        )
