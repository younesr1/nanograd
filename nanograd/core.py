#!/usr/bin/python3
from enum import Enum
import numpy as np

class Operation(Enum):
    addition = '+'
    product = '*'
    relu = 'relu'
    none = ''

class Scalar:
    def __init__(self, value, parents=(), operation=Operation.none):
        self.value = value
        self.grad = 0
        assert len(parents) == len(set(parents))
        assert isinstance(parents, tuple)
        self.parents = parents
        self.operation = operation

    def __add__(self, b):
        assert isinstance(b, Scalar)
        return Scalar(self.value+b.value, (self, b), Operation.addition)

    def __mul__(self, b):
        assert isinstance(b, Scalar)
        return Scalar(self.value*b.value, (self, b), Operation.product)

    def relu(self):
        val = max(0, self.value)
        return Scalar(val, (self,), Operation.relu)

    def zero_grad(self):
        self.grad = 0

    def backward(self, internal = False):
        if not internal:
            self.grad = 1
        if self.operation == Operation.product:
            assert len(self.parents) == 2
            self.parents[0].grad += self.parents[1].value * self.grad
            self.parents[1].grad += self.parents[0].value * self.grad
        elif self.operation == Operation.addition:
            assert len(self.parents) == 2
            self.parents[0].grad += 1 * self.grad
            self.parents[1].grad += 1 * self.grad
        elif self.operation == Operation.relu:
            assert len(self.parents) == 1
            self.parents[0].grad += int(self.parents[0].value > 0) * self.grad
        elif self.operation == Operation.none:
            assert len(self.parents) == 0
        else:
            assert False and "operation not recognized"

        for parent in self.parents:
            parent.backward(True)
 
    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        return f"<{module}.{qualname} object with value {self.value} at {hex(id(self))}>"


class Neuron:
    def __init__(self, input_count, activation_func):
        # init with random float in range [-10, 10)
        # it makes sense that the weight and bias have no parents!
        # picture the computational graph!
        self.weights = np.array([Scalar(np.random.ranf() * 2 * 10 -10) for _ in range(input_count)])
        self.bias = Scalar(0)
        assert activation_func in [Operation.relu, Operation.none]
        self.activation_func = activation_func

    def __call__(self,x):
        assert x.shape == self.weights.shape and all(isinstance(xi, Scalar) for xi in x)
        activation = np.dot(self.weights, x) + self.bias
        return activation.relu() if self.activation_func is Operation.relu else activation

    def zero_grad(self):
        for scalar in np.concatenate(self.weights, np.array([self.bias])): scalar.zero_grad()

    def count_parameters(self):
        return len(self.weights + [self.bias])

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        return (f"<{module}.{qualname} object with {self.count_parameters()} params and "
                f"{self.activation_func.name} activation function  at {hex(id(self))}>")


class LinearLayer:
    def __init__(self, inputs_per_neuron, number_of_neurons, activation_func):
        self.inputs_per_neuron = inputs_per_neuron
        self.number_of_neurons = number_of_neurons
        self.activation_func = activation_func
        self.neurons = np.array([Neuron(inputs_per_neuron, activation_func) for _ in range(number_of_neurons)])

    def __call__(self,x):
        assert len(x) == self.inputs_per_neuron and all(isinstance(xi, Scalar) for xi in x)
        ret = np.array([neuron(x) for neuron in self.neurons])
        return ret if len(ret) > 1 else ret[0]

    def zero_grad(self):
        for neuron in self.neurons: neuron.zero_grad()

    def count_parameters(self):
        return sum(neuron.count_parameters() for neuron in self.neurons)

    def weights(self):
        return np.array([neuron.weight for neuron in self.neurons])

    def biases(self):
        return np.array([neuron.bias for neuron in self.neurons])

    def __repr__(self):
        return (f"<{module}.{qualname} object with {self.number_of_neurons} neurons each with"
                f"{self.inputs_per_neuron} inputs and {self.activation_func.name}"
                f"activation function  at {hex(id(self))}>")


class Network:
    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.l1 = LinearLayer(2, 10, Operation.relu)
        self.l2 = LinearLayer(10, 10, Operation.relu)
        self.l3 = LinearLayer(10, 1, Operation.none)
        for l,w,b in [(self.l1,w1,b1),(self.l2,w2,b2),(self.l3,w3,b3)]:
            assert all(isinstance(bi, Scalar) for bi in b)
            assert all(isinstance(wij, Scalar) for row in w for wij in row)
            assert weights.shape == w.shape
            assert biases.shape() == b.shape
            weights = l.weights()
            weights = w
            biases = l.biases()
            biases = b

    def __call__(self,x):
        assert len(x) == 2 and all(isinstance(xi, Scalar) for xi in x)
        out1 = self.l1(x)
        out2 = self.l2(out1)
        out3 = self.l3(out2)
        return out3

    def zero_grad(self):
        self.l1.zero_grad()
        self.l2.zero_grad()
        self.l3.zero_grad()

    def count_parameters(self):
        return sum(layer.count_parameters() for layer in [self.l1, self.l2, self.l3])
        






