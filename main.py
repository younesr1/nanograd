#!/usr/bin/python3
from nanograd.network import Network
from nanograd.tensor import Tensor
import pickle
import numpy as np
import matplotlib.pyplot as plt

def loss(y_hat: float, y: float) -> float:
    assert isinstance(y_hat, float) and isinstance(y, float)
    return np.square(y_hat-y)/2

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

    inputs = [Tensor(data=np.reshape(i, (-1,1))) for i in inputs]
    w1 = Tensor(data=w1)
    w2 = Tensor(data=w2)
    w3 = Tensor(data=w3)
    b1 = Tensor(data=np.reshape(b1, (-1,1)))
    b2 = Tensor(data=np.reshape(b2, (-1,1)))
    b3 = Tensor(data=np.reshape(b3, (-1,1)))
    f.close()
    return inputs, targets, w1, b1, w2, b2, w3, b3
    
def main():
    inputs, targets, w1_naught, b1_naught, w2_naught, b2_naught, w3_naught, b3_naught = get_data()
    assert len(inputs) == len(targets)

    nn = Network(w1_naught,b1_naught,w2_naught,b2_naught,w3_naught,b3_naught)
    print("network shapes", nn.get_shapes())

    losses = []
    epochs = 5
    learning_rate = 1/100
    for i in range(epochs):
        epoch_loss = 0
        w1_grad_sum = 
        for x,y in zip(inputs, targets):
            y_hat = nn.forward(x)
            assert y_hat.shape() == (1,1)
            y_hat = y_hat.data[0,0]
            epoch_loss += loss(y_hat,y)
            nn.backward(epoch_loss)
            nn.update
        epoch_loss = epoch_loss / len(inputs)
        losses.append(epoch_loss)
    #plt.plot(losses)
    #plt.show()

if __name__ == '__main__':
    main()
