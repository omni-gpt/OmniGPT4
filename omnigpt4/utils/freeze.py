import torch.nn as nn


def disabled_train(self, mode: bool = True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.train = disabled_train
