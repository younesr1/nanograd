#!/usr/bin/python3
import pickle
import numpy as np
from nanograd.core import Scalar, Network
from nanograd.visualizer import Viz

def get_data():
    f = open("assignment-one-test-parameters.pkl", "rb")                                             
    assignment_data = pickle.load(f)

    convert = lambda arr: np.array([Scalar(i) for i in arr])
    inputs = np.array([convert(row) for row in assignment_data['inputs']])
    targets = np.array([Scalar(t) for t in assignment_data['targets']])

    w1 = np.array([convert(row) for row in assignment_data['w1']])
    w2 = np.array([convert(row) for row in assignment_data['w2']])
    w3 = np.array([convert(row) for row in assignment_data['w3']])


    b1 = np.array([Scalar(b) for b in assignment_data['b1']])
    b2 = np.array([Scalar(b) for b in assignment_data['b2']])
    b3 = np.array([Scalar(b) for b in assignment_data['b3']])

    before = [assignment_data['inputs'], assignment_data['targets'], assignment_data['w1'], assignment_data['b1'], assignment_data['w2'], assignment_data['b2'], assignment_data['w3'], assignment_data['b3']]
    after = [inputs, targets, w1, b1, w2, b2, w3, b3]
    for b,a in zip(before,after):
        assert b.shape == a.shape
        assert all(bef == aft.value for bef,aft in zip(b.flatten(),a.flatten()))

    return inputs, targets, w1, b1, w2, b2, w3, b3
    
def main():
    inputs,targets,w1,b1,w2,b2,w3,b3 = get_data()
    nn = Network(w1,b1,w2,b2,w3,b3)
    x = inputs
    y = targets 
    epochs = 5
    lr = 0.01
    losses = []
    for i in range(epochs):
        epoch_loss = 0
        #print(nn.l1.weights())
        for xi,yi in zip(x,y):
            y_hat = nn(xi)
            loss = y_hat.regression(yi)
            epoch_loss += loss.value 
            nn.zero_grad() # younes todo does zero grad actually work?
            loss.backward()
            nn.update_parameters(lr)# younes todo should only do 1 update per epoch
        losses.append(epoch_loss/len(x))
    print(losses)

if __name__ == '__main__':
    main()
