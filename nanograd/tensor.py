#!/usr/bin/python3
import numpy as np

class Tensor:
    def __init__(self, shape = None, data = None):
        assert shape is not None or data is not None
        if data is not None:
            self.data = data
        else:
            assert isinstance(shape, tuple)
            self.data = np.zeros(shape)

    def shape(self):
        return self.data.shape

    def set_zero(self):
        self.data = np.zeros(self.shape())

    def set_identity(self):
        assert self.shape()[0] == self.shape()[1]
        self.data = np.identity(self.shape()[0])

    def set_random(self):
        self.data = np.random.random_sample(self.shape())

    # Adds tensors as this + b
    def __add__(self, b: '__class__') -> '__class__':
        assert isinstance(b,Tensor)
        assert self.shape() == b.shape() 
        assert len(self.shape()) == 2 and "Binary Ops only supported for 2nd order Tensors"
        return Tensor(data=self.data+b.data)

    # Subtracts tensors as this - b
    def __sub__(self, b: '__class__') -> '__class__':
        assert isinstance(b,Tensor)
        assert self.shape() == b.shape()
        assert len(self.shape()) == 2 and "Binary Ops only upported for 2nd order Tensors"
        return Tensor(data=self.data-b.data)

    # implements tensor multiplcation as this * b
    def __mul__(self, b: '__class__') -> '__class__': 
        assert isinstance(b,Tensor)
        assert len(self.shape()) == len(b.shape()) == 2 and "Binary Ops only supported for 2nd order Tensors"
        (m,n),(k,l) = (self.shape(), b.shape())
        assert n == k
        return Tensor(data = np.matmul(self.data, b.data))

    # implements equality check of tensors
    def __eq__(self, b: '__class__') -> bool:
        assert self.shape() == b.shape()
        return (self.data == b.data).all()

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        return f"<{module}.{qualname} object of shape {self.data.shape} at {hex(id(self))}>"

    def __str__(self):
        return self.__repr__() + '\n' + self.data.__str__()



