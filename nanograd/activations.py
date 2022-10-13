#!/usr/bin/python3
from nanograd.tensor import Tensor
import numpy as np

def relu(x: Tensor) -> Tensor:
    data = np.vectorize(lambda a : max(0,a))(x.data)
    return Tensor(data=data)

