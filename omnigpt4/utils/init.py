from contextlib import contextmanager
from typing import Iterable, Type, Protocol, Optional

import torch.nn as nn


class SupportsResetParameters(Protocol):
    def reset_parameters(self):
        ...


@contextmanager
def no_init(
    enabled: bool = True,
    module_classes: Optional[Iterable[Type[SupportsResetParameters]]] = None
):
    if enabled:
        if module_classes is None:
            module_classes = [nn.Linear, nn.Conv2d, nn.Embedding]

        saved_fn = {m: getattr(m, "reset_parameters", None) for m in module_classes}

        def no_op(_):
            ...

        for m in saved_fn.keys(): m.reset_parameters = no_op

    try:
        yield
    finally:
        if enabled:
            for m, init in saved_fn.items():
                del m.reset_parameters
                if init is not None:
                    m.reset_parameters = init
