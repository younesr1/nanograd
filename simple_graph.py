#!/usr/bin/python3
from nanograd.core import Scalar

def main():
    a = Scalar(5)
    b = Scalar(7)
    c = a + b
    print(c)
    for v in [a,b,c]:
        v.zero_grad()
    c.backward()
    print(a.grad, b.grad, c.grad)
    

if __name__ == '__main__':
    main()
