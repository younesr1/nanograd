#!/usr/bin/python3
from nanograd.tensor import Tensor

class FullyConnected:
    def __init__(self, weights: Tensor, biases: Tensor):
        assert len(weights.shape()) == len(biases.shape()) == 2
        assert biases.shape()[1] == 1
        self.weights = weights
        self.biases = biases
        self.weights_grad = Tensor(shape=self.weights.shape())
        self.biases_grad = Tensor(shape=self.biases.shape())

    def __call__(self, x: Tensor) -> Tensor:
       return self.weights*x+self.biases 

    def zero_grad(self) -> None:
        self.weights_grad.set_zero()
        self.biases_grad.set_zero()


