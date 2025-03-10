import numpy as np
import random
import gin
from typing import Dict, Tuple
import numpy as np
import torch


class Transform():

    def __init__(self, sr: int | None, name: str | None) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x):
        raise NotImplementedError()

@gin.configurable(module="transforms")
class RandomApply(Transform):
    """
    Apply transform with probability p
    """
    def __init__(self, transform, p=.5, batchwise: bool = True):
        self.transform = transform
        self.batchwise = batchwise
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.batchwise:
            random = torch.rand(x.shape[0] + (1,) * (x.ndim - 1)).expand_as(x)
            x = torch.where(random < self.p, self.transform(x), x)
        else:
            if random.random() < self.p:
                x = self.transform(x)
        return x

@gin.configurable(module="transforms")
class Compose(Transform):
    """
    Apply a list of transform sequentially
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        for elm in self.transform_list:
            x = elm(x)
        return x
