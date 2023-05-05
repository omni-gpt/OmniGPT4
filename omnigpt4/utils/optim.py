from typing import List, Tuple, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.,
        warmup_init_lr: float = 0.,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.num_warmup_steps = num_warmup_steps
        self.warmup_init_lr = warmup_init_lr

        super().__init__(
            optimizer=optimizer,
            T_max=num_training_steps,
            eta_min=min_lr,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def _get_warmup_stage_lr(self, init_lr: float):
        return min(
            init_lr,
            self.warmup_init_lr + (init_lr - self.warmup_init_lr) * self.last_epoch / max(self.num_warmup_steps, 1)
        )

    def get_lr(self) -> float:
        if self.last_epoch < self.num_warmup_steps:
            lr = [
                self._get_warmup_stage_lr(group["initial_lr"])
                for group in self.optimizer.param_groups
            ]
        else:
            self.last_epoch -= self.num_warmup_steps
            lr = super().get_lr()
            self.last_epoch += self.num_warmup_steps

        return lr
