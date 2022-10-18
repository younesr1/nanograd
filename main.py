#!/usr/bin/python3
import pickle
import numpy as np

def get_data():
    f = open("assignment-one-test-parameters.pkl", "rb")                                             
    assignment_data = pickle.load(f)
    inputs = assignment_data["inputs"]
    targets = assignment_data["targets"]
    w1 = assignment_data["w1"]
    w2 = assignment_data["w2"]
    w3 = assignment_data["w3"]
    b1 = assignment_data["b1"]
    b2 = assignment_data["b2"]
    b3 = assignment_data["b3"]

    return inputs, targets, w1, b1, w2, b2, w3, b3
    
def main():
    pass

if __name__ == '__main__':
    main()
