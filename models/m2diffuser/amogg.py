import torch
from typing import Callable, Dict, Optional, Union


Tensor = torch.Tensor
ObjectiveFn = Callable[[Tensor, Dict], Tensor]


class AMOGGSampler:
    """Adaptive Multi-Objective Gradient Guidance sampler.

    This module implements:
      1) Time-aware scheduling:
         lambda(t) = sigmoid(kappa * (u - 1/2)), u = t / (T - 1)
      2) Conflict-averse projection:
         if <g1, g2> < 0, project g2 to the normal plane of g1
      3) Guidance injection:
         g = lambda * g1 + (1 - lambda) * g2
         guidance = Sigma_t * g (optional)
    """

    def __init__(self, kappa: float = 10.0, eps: float = 1e-8, detach_output: bool = True) -> None:
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.detach_output = bool(detach_output)

    @staticmethod
    def _reduce_objective(obj: Tensor) -> Tensor:
        if obj.ndim == 0:
            return obj
        if obj.ndim == 1:
            return obj.sum()
        raise ValueError(f"Objective must be scalar or [B], got shape={tuple(obj.shape)}")

    def _compute_grad(self, x: Tensor, data: Dict, objective_fn: ObjectiveFn) -> Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            objective = objective_fn(x_in, data)
            if not torch.is_tensor(objective):
                objective = torch.as_tensor(objective, dtype=x_in.dtype, device=x_in.device)
            objective = self._reduce_objective(objective)
            grad = torch.autograd.grad(
                outputs=objective,
                inputs=x_in,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]

        return grad

    def _time_weight(self, t: Union[int, Tensor], T: int, ref: Tensor) -> Tensor:
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}.")

        denom = float(max(T - 1, 1))

        if torch.is_tensor(t):
            t_tensor = t.to(device=ref.device, dtype=ref.dtype)
        else:
            t_tensor = torch.tensor(float(t), device=ref.device, dtype=ref.dtype)

        u = t_tensor / denom
        lam = torch.sigmoid(self.kappa * (u - 0.5))

        if lam.ndim == 0:
            lam = lam.view(1, *([1] * (ref.ndim - 1)))
        elif lam.ndim == 1:
            lam = lam.view(-1, *([1] * (ref.ndim - 1)))
        else:
            raise ValueError(f"t must be scalar or [B], got shape={tuple(lam.shape)}")

        return lam

    def _project_conflict_averse(self, g1: Tensor, g2: Tensor) -> Tensor:
        if g1.shape != g2.shape:
            raise ValueError(f"g1 and g2 must have same shape, got {g1.shape} vs {g2.shape}")
        if g1.ndim < 2:
            raise ValueError(f"Expected at least 2 dims [B, ...], got ndim={g1.ndim}")

        reduce_dims = tuple(range(1, g1.ndim))
        dot_12 = torch.sum(g1 * g2, dim=reduce_dims, keepdim=True)
        norm_1_sq = torch.sum(g1 * g1, dim=reduce_dims, keepdim=True).clamp_min(self.eps)

        g2_projected = g2 - (dot_12 / norm_1_sq) * g1
        conflict_mask = dot_12 < 0
        g2_safe = torch.where(conflict_mask, g2_projected, g2)

        return g2_safe

    def compute_guidance(
        self,
        x_prev_joint: Tensor,
        data: Dict,
        t: Union[int, Tensor],
        T: int,
        task_objective_fn: ObjectiveFn,
        collision_objective_fn: ObjectiveFn,
        sigma_t: Optional[Tensor] = None,
        task_scale: float = 1.0,
        collision_scale: float = 1.0,
    ) -> Tensor:
        """Compute AMOGG guidance for current denoising step.

        Args:
            x_prev_joint: Trajectory tensor at t-1 in joint space, shape [B, H, D].
            data: Auxiliary dict consumed by objective functions.
            t: Current timestep (scalar int or tensor [B]).
            T: Total diffusion steps.
            task_objective_fn: Function c_task(x, data) to maximize.
            collision_objective_fn: Function c_collision(x, data) to maximize.
            sigma_t: Optional variance tensor Sigma_t for guidance injection.
            task_scale: Scale factor for task gradient.
            collision_scale: Scale factor for collision gradient.
        Returns:
            Guidance tensor aligned with x_prev_joint shape.
        """
        g1 = self._compute_grad(x_prev_joint, data, task_objective_fn)
        g2 = self._compute_grad(x_prev_joint, data, collision_objective_fn)

        g1 = float(task_scale) * g1
        g2 = float(collision_scale) * g2

        g2 = self._project_conflict_averse(g1, g2)

        lam = self._time_weight(t, T, x_prev_joint)
        g = lam * g1 + (1.0 - lam) * g2
        guidance = g if sigma_t is None else sigma_t * g

        if self.detach_output:
            guidance = guidance.detach()

        return guidance
