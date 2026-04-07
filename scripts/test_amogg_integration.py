"""
AMOGG implementation smoke tests.

Run:
    python scripts/test_amogg_integration.py

This script validates:
1) AMOGG core math (time-aware weighting + conflict-averse projection + Sigma_t injection)
2) DDPM.p_sample() integration path (AMOGG branch is actually used when enabled)

The script is intentionally self-contained and avoids full project runtime dependencies.
It dynamically loads modules from source files and installs lightweight stubs for
missing training-time packages.
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_close(a: torch.Tensor, b: torch.Tensor, msg: str, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"{msg} (max_abs_err={max_abs:.6e})")


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stub_modules() -> None:
    """Install minimal stubs so ddpm.py can be imported without full environment."""

    # Stub pytorch_lightning with minimal LightningModule behavior.
    if "pytorch_lightning" not in sys.modules:
        pl_mod = types.ModuleType("pytorch_lightning")

        class LightningModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def log(self, *args: Any, **kwargs: Any) -> None:
                return None

        pl_mod.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = pl_mod

    # Stub omegaconf DictConfig type for annotations.
    if "omegaconf" not in sys.modules:
        oc_mod = types.ModuleType("omegaconf")

        class DictConfig(dict):
            pass

        oc_mod.DictConfig = DictConfig
        sys.modules["omegaconf"] = oc_mod

    # Prepare package namespaces so relative imports resolve without executing models/__init__.py.
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = [str(REPO_ROOT / "models")]
    sys.modules["models"] = models_pkg

    models_m2_pkg = types.ModuleType("models.m2diffuser")
    models_m2_pkg.__path__ = [str(REPO_ROOT / "models" / "m2diffuser")]
    sys.modules["models.m2diffuser"] = models_m2_pkg

    models_opt_pkg = types.ModuleType("models.optimizer")
    models_opt_pkg.__path__ = [str(REPO_ROOT / "models" / "optimizer")]
    sys.modules["models.optimizer"] = models_opt_pkg

    models_plan_pkg = types.ModuleType("models.planner")
    models_plan_pkg.__path__ = [str(REPO_ROOT / "models" / "planner")]
    sys.modules["models.planner"] = models_plan_pkg

    # Stub registry used by @DIFFUSER.register decorator.
    base_mod = types.ModuleType("models.base")

    class _Registry:
        def __init__(self, name: str) -> None:
            self.name = name
            self._store: Dict[str, Any] = {}

        def register(self):
            def _decorator(cls):
                self._store[cls.__name__] = cls
                return cls

            return _decorator

        def get(self, key: str):
            return self._store[key]

    base_mod.DIFFUSER = _Registry("Diffuser")
    sys.modules["models.base"] = base_mod

    # Stub parent classes for type annotations.
    optimizer_mod = types.ModuleType("models.optimizer.optimizer")

    class Optimizer:
        pass

    optimizer_mod.Optimizer = Optimizer
    sys.modules["models.optimizer.optimizer"] = optimizer_mod

    planner_mod = types.ModuleType("models.planner.planner")

    class Planner:
        pass

    planner_mod.Planner = Planner
    sys.modules["models.planner.planner"] = planner_mod


def _to_cfg(obj: Any) -> Any:
    """Convert nested dict to attribute-access cfg object."""

    class AttrDict(dict):
        def __getattr__(self, item: str) -> Any:
            if item in self:
                return self[item]
            raise AttributeError(item)

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    if isinstance(obj, dict):
        cfg = AttrDict()
        for k, v in obj.items():
            cfg[k] = _to_cfg(v)
        return cfg
    if isinstance(obj, list):
        return [_to_cfg(v) for v in obj]
    return obj


def test_amogg_core_math() -> None:
    amogg_mod = _load_module("amogg_unit", REPO_ROOT / "models" / "m2diffuser" / "amogg.py")
    AMOGGSampler = amogg_mod.AMOGGSampler

    sampler = AMOGGSampler(kappa=20.0, eps=1e-8, detach_output=True)
    # B=2, H=1, D=2
    x = torch.zeros(2, 1, 2, dtype=torch.float32)

    c1 = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32)   # g1 for both batches
    c2 = torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32)  # batch0 conflict, batch1 aligned

    def task_obj(x_in: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (x_in * data["c1"]).sum(dim=(1, 2))

    def collision_obj(x_in: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (x_in * data["c2"]).sum(dim=(1, 2))

    data = {"c1": c1, "c2": c2}

    # Use midpoint t so lambda ~= 0.5
    guidance = sampler.compute_guidance(
        x_prev_joint=x,
        data=data,
        t=2,
        T=5,
        task_objective_fn=task_obj,
        collision_objective_fn=collision_obj,
        sigma_t=torch.ones_like(x),
    )

    # batch0: g2 conflicts with g1 -> projected g2 becomes 0 -> g = 0.5 * g1
    # batch1: no conflict and g2 == g1 -> g = g1
    expected = torch.tensor([[[0.5, 0.0]], [[1.0, 0.0]]], dtype=torch.float32)
    _assert_close(guidance, expected, "AMOGG conflict projection or weighted combination is incorrect.")

    # Verify detach behavior
    if guidance.requires_grad:
        raise AssertionError("Guidance should be detached when detach_output=True.")

    # Verify schedule direction: early < 0.5, late > 0.5
    lam_early = sampler._time_weight(t=0, T=5, ref=x).reshape(-1)[0].item()
    lam_late = sampler._time_weight(t=4, T=5, ref=x).reshape(-1)[0].item()
    if not (lam_early < 0.5 < lam_late):
        raise AssertionError(f"Time-aware lambda monotonicity failed: early={lam_early:.4f}, late={lam_late:.4f}")

    print("[PASS] AMOGG core math test")


def test_ddpm_amogg_integration_smoke() -> None:
    _install_stub_modules()

    # Load real schedule + AMOGG + DDPM source files into stub package namespace.
    _load_module("models.m2diffuser.schedule", REPO_ROOT / "models" / "m2diffuser" / "schedule.py")
    _load_module("models.m2diffuser.amogg", REPO_ROOT / "models" / "m2diffuser" / "amogg.py")
    ddpm_mod = _load_module("models.m2diffuser.ddpm", REPO_ROOT / "models" / "m2diffuser" / "ddpm.py")
    DDPM = ddpm_mod.DDPM

    class DummyEpsModel(nn.Module):
        def condition(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.zeros((data["x"].shape[0], 1), dtype=data["x"].dtype, device=data["x"].device)

        def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            # Predict zero noise to make mean behavior easier to reason about.
            return torch.zeros_like(x_t)

    class DummyPlanner:
        def objective(self, x: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
            return (x * data["task_coeff"]).sum(dim=(1, 2))

        def gradient(self, x: torch.Tensor, data: Dict[str, torch.Tensor], variance: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Fallback planner.gradient should not be called when AMOGG is active.")

    class DummyOptimizer:
        def collision_objective(self, x: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
            return (x * data["collision_coeff"]).sum(dim=(1, 2))

        def gradient(self, x: torch.Tensor, data: Dict[str, torch.Tensor], variance: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Fallback optimizer.gradient should not be called when AMOGG is active.")

    cfg = _to_cfg(
        {
            "timesteps": 8,
            "schedule_cfg": {"beta": [0.0001, 0.02], "beta_schedule": "linear", "s": 0.008},
            "rand_t_type": "half",
            "loss_type": "l2",
            "lr": 1e-4,
            "sample": {
                "converage": {"optimization": True, "planning": True, "ksteps": 1},
                "fine_tune": {"optimization": True, "planning": True, "timesteps": 2, "ksteps": 1},
                "amogg": {
                    "enabled": True,
                    "kappa": 10.0,
                    "eps": 1e-8,
                    "use_variance": True,
                    "task_scale": 1.0,
                    "collision_scale": 1.0,
                    "detach_output": True,
                },
            },
        }
    )

    model = DDPM(eps_model=DummyEpsModel(), cfg=cfg, has_obser=False)
    model.set_optimizer(DummyOptimizer())
    model.set_planner(DummyPlanner())

    B, H, D = 2, 3, 2
    x_t = torch.randn(B, H, D, dtype=torch.float32)
    data = {
        "x": x_t.clone(),
        "task_coeff": torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32).expand(B, H, D),
        "collision_coeff": torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32).expand(B, H, D),
    }

    # t=0 => deterministic sample output without random noise term.
    t = 0
    with torch.no_grad():
        batch_timestep = torch.full((B,), t, device=model.device, dtype=torch.long)
        cond = model.eps_model.condition(data)
        base_mean, model_variance, _ = model.p_mean_variance(x_t, batch_timestep, cond)

    expected_guidance = model.amogg.compute_guidance(
        x_prev_joint=base_mean,
        data=data,
        t=t,
        T=model.timesteps,
        task_objective_fn=model.planner.objective,
        collision_objective_fn=model.optimizer.collision_objective,
        sigma_t=model_variance,
        task_scale=model.amogg_task_scale,
        collision_scale=model.amogg_collision_scale,
    )
    expected = base_mean + expected_guidance
    actual = model.p_sample(x_t, t=t, data=data, opt=True, plan=True)

    _assert_close(actual, expected, "DDPM p_sample AMOGG branch output mismatch.")
    print("[PASS] DDPM AMOGG integration smoke test")


def main() -> None:
    torch.manual_seed(0)
    test_amogg_core_math()
    test_ddpm_amogg_integration_smoke()
    print("\nAll AMOGG tests passed.")


if __name__ == "__main__":
    main()
