#!/usr/bin/python3
from nanograd.core import Scalar
from nanograd.visualizer import Viz

def main():
    a = Scalar(-5)
    b = Scalar(7)
    c = a * b
    d = c.relu()
    e = Scalar(2)
    f = Scalar(3)
    g = e + f
    h = g.relu()
    loss = h+d
    loss.backward()
    viz = Viz(loss)
    viz.view()    
    

if __name__ == '__main__':
    main()
