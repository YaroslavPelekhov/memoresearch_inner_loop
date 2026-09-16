import torch


@torch.jit.script
def z_loss_fn(logits: torch.Tensor, eps: float):
    return eps * torch.square(torch.logsumexp(logits, dim=-1))

@torch.jit.script
def hidden_z_loss_fn(hidden_states: torch.Tensor, eps: float):
    return eps * torch.square(torch.logsumexp(hidden_states.abs(), dim=-1))