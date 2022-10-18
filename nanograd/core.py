#!/usr/bin/python3
from enum import Enum

class Operation(Enum):
    addition = 0
    product = 1
    relu = 2
    none = 3

class Scalar:
    def __init__(self, value, parents=(), operation=Operation.none):
        self.value = value
        self.grad = 0
        self.parents = parents # younes todo parents should be a set but its not subscriptable
        self.operation = operation

    def __add__(self, b):
        assert isinstance(b, Scalar)
        return Scalar(self.value+b.value, (self, b), Operation.addition)

    def __mul__(self, b):
        assert isinstance(b, Scalar)
        return Scalar(self.value*b.value, (self, b), Operation.product)

    def relu(self):
        return max(0, self.value)

    def zero_grad(self):
        self.grad = 0

    def backward(self, prev_grad = None):
        if prev_grad is None:
            prev_grad = 1
            self.grad = 1
        if self.operation == Operation.product:
            assert len(self.parents) == 2
            self.parents[0].grad += self.parents[1].value * prev_grad
            self.parents[1].grad += self.parents[0].value * prev_grad
        elif self.operation == Operation.addition:
            assert len(self.parents) == 2
            self.parents[0].grad += 1 * prev_grad
            self.parents[1].grad += 1 * prev_grad
        elif self.operation == Operation.relu:
            assert len(self.parents) == 1
            assert False
        elif self.operation == Operation.none:
            assert len(self.parents) == 0
        else:
            assert False and "operation not recognized"

        for parent in self.parents:
            parent.backward(self.grad)
         

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        return f"<{module}.{qualname} object with value {self.value} at {hex(id(self))}>"




