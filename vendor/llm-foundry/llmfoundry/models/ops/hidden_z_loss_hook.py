from typing import Optional, Tuple

import torch

from llmfoundry.models.ops.zloss import hidden_z_loss_fn


class HiddenZLossRegularizer(torch.autograd.Function):
    """Pass-through autograd Function that splices a hidden z-loss gradient
    into the graph without keeping logsumexp/square/mean intermediates alive.

    forward(x, coef):
        Computes the scalar `loss = hidden_z_loss_fn(x, coef).mean()` under
        `torch.no_grad()` for logging (no autograd intermediates are saved).
        Stores a reference to `x` via `save_for_backward` — `x` is already
        kept alive by the downstream post-norm, so this does not retain any
        extra GPU storage. Returns `(x, loss_detached)`.

    backward(grad_y, _):
        Reconstructs a small autograd graph in a scoped `torch.enable_grad()`
        from the saved `x`, recomputes `loss`, and obtains `d(loss)/dx` via
        `torch.autograd.grad`. The total gradient w.r.t. `x` is
        `grad_y + d(loss)/dx` — equivalent to having added `loss` to the
        training objective, but with zero extra forward-activation memory.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, coef: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x)
        ctx.coef = coef
        with torch.no_grad():
            loss_val = hidden_z_loss_fn(x, coef).mean()
        # `loss_val` is for logging only — no gradient should flow back through
        # it from the caller side (the regularizer's contribution to d/dx is
        # spliced in via the `grad_y + g_reg` rule in backward, not through
        # this output).
        ctx.mark_non_differentiable(loss_val)
        return x, loss_val

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, _grad_loss_val):
        (x,) = ctx.saved_tensors
        coef = ctx.coef
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            loss = hidden_z_loss_fn(x_req, coef).mean()
            (g_reg,) = torch.autograd.grad(loss, x_req)
        if grad_y is None:
            return g_reg, None
        return grad_y + g_reg, None


def apply_hidden_z_loss_hook(
    hidden_states: torch.Tensor,
    coef: float,
    *,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Splice hidden z-loss gradient into the autograd graph at this point
    without storing any forward intermediates (logsumexp/square/mean are
    recomputed in backward). Returns `(hidden_states, loss_scalar_or_None)`.
    In eval / `coef == 0.0` this is a no-op.
    """
    if not training or coef == 0.0:
        return hidden_states, None
    hidden_states, loss_val = HiddenZLossRegularizer.apply(hidden_states, coef)
    return hidden_states, loss_val
