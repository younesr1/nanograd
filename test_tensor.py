#!/usr/bin/python3
from nanograd.tensor import Tensor
import numpy as np

def test_addition():
    a = Tensor(data=np.array([[1,2,3],[4,5,6],[7,8,9]]))
    b = Tensor(data=np.array([[10,20,30],[40,50,60],[70,80,90]]))
    assert (a+b) == Tensor(data=np.array([[11,22,33],[44,55,66],[77,88,99]]))
    assert (a+b) == (b+a)


def test_subtraction():
    a = Tensor(data=np.array([[1,2,3],[4,5,6],[7,8,9]]))
    b = Tensor(data=np.array([[10,20,30],[40,50,60],[70,80,90]]))
    assert (b-a) == Tensor(data=np.array([[9, 18, 27], [36, 45, 54], [63, 72, 81]]))

def main():
    print("testing tensor binary operations")
    test_addition()
    test_subtraction()
    print("done")

if __name__ == "__main__":
    main()
