import numpy as np
from collections import UserList
from typing import List
import random
import gin
import numpy as np
import torch
from enum import Enum

class AcidsTransformException(Exception):
    pass

class TransformInput(Enum): 
    none = 0
    numpy = 1
    torch = 2

default_cast_table = {
    (TransformInput.numpy, TransformInput.numpy): lambda x: x,
    (TransformInput.numpy, TransformInput.torch): lambda x: torch.from_numpy(x).float(),
    (TransformInput.torch, TransformInput.numpy): lambda x: x.numpy(), 
    (TransformInput.torch, TransformInput.torch): lambda x: x
}

class Transform():
    allow_random: bool = True
    input_types = TransformInput
    takes_as_input = TransformInput.none
    cast_table = default_cast_table
    def __init__(self,
                 sr: int | None = None, 
                 name: str | None = None, 
                 p: float | None = None, 
                 rand_batchwise: bool = True) -> None:
        """Abstract class for every transform

        Args:
            sr (int | None, optional): Input sampling rate. Defaults to None.
            name (str | None, optional): Optional transform name. Defaults to None.
            p (float | None, optional): Optional probability (must allow random). Defaults to None.
            rand_batchwise (bool, optional): Is randomness batchwise or indexwise. Defaults to True.
        """
        self.sr = sr
        self.name = name
        self.p = p
        self.rand_batchwise = rand_batchwise
        self.rng = np.random.default_rng(12345)

    def type_hash(self, data):
        if isinstance(data, np.ndarray):
            return self.input_types.numpy
        elif torch.is_tensor(data):
            return self.input_types.torch
        else: 
            return self.input_types.none

    def _parse_arg(self, arg):
        if self.takes_as_input == self.input_types.none:
            return arg
        else:
            return self.cast_table[self.type_hash(arg), self.takes_as_input](arg)

    def _parse_output(self, output, input):
        out = self.cast_table[(self.type_hash(output), self.type_hash(input))](output)
        if self.type_hash(out) == self.input_types.torch:
            out = out.to(input)
        return out

    def apply(self, x: np.ndarray):
        return x

    def apply_random(self, x):
        if not self.allow_random:
            raise AcidsTransformException("transform %s does not allow random."%type(self).__name__)
        if self.rand_batchwise:
            if torch.is_tensor(x):
                rnm = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1)).expand_as(x)
                x = torch.where(rnm < self.p, self.apply(x), x)
            else:
                rnm = np.broadcast_to(self.rng.uniform(size=(x.shape[0],) + (1,) * (x.ndim - 1)), x.shape)
                x = np.where(rnm < self.p, self.apply(x), x) 
        else:
            if random.random() < self.p:
                x = self.apply(x)
        return x

    def __call__(self, *args, **kwargs):
        data_in = args[0]
        args = tuple(map(self._parse_arg, args))
        if (self.p is None) or (not self.allow_random):
            out = self.apply(*args, **kwargs)
        else:
            out = self.apply_random(*args, **kwargs)
        out = self._parse_output(out, data_in)
        return out


@gin.configurable(module="transforms")
class Compose(UserList):
    def __init__(self, *transforms: List[Transform]):
        """Compose sequentially applies conteined transforms. Subclass of UserLit, such that most methods like append, extend, 
        etc, are handled. 

        Args:
            transform_list (List[Transform]): list of transforms to apply.
        """
        super.__init__(*transforms)

    def __call__(self, x: np.ndarray):
        for elm in self:
            x = elm(x)
        return x
