#!/usr/bin/python3
from nanograd.layer import FullyConnected
from nanograd.tensor import Tensor
from nanograd.activations import relu

class Network:
    def __init__(self,w1,b1,w2,b2,w3,b3):
        self.l1 = FullyConnected(w1,b1)
        self.l2 = FullyConnected(w2,b2)
        self.l3 = FullyConnected(w3,b3)

    def forward(self, x: Tensor) -> Tensor:
        self.l1.zero_grad()
        self.l2.zero_grad()
        self.l3.zero_grad()
        self.l1_activation = self.l1(x)
        out1 = relu(self.l1_activation)
        self.l2_activation = self.l2(out1)
        out2 = relu(self.l2_activation)
        out3 = self.l3(out2)
        return out3

    def backward(self, loss) -> None:
        # fill .weights_grad & .biases_grad for each layer

    def get_shapes(self):
        ret = {}
        for index, layer in enumerate([self.l1, self.l2, self.l3]):
            ret[f"layer{index+1}"] = {"weights": layer.weights.shape(), "biases": layer.biases.shape()}
        return ret


