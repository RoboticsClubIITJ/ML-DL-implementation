from .optimizers import GradientDescent
from .utils import generate_weights
import numpy as np
import pickle


class LinearRegression():

    def fit(self, X, Y, optimizer=GradientDescent, epochs=25, zeros=False):

        self.weights = generate_weights(X.shape[1], 1, zeros=zeros)

        print("Starting training with loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        for epoch in range(1, epochs+1):
            print("======================================")
            self.weights = optimizer.iterate(X, Y, self.weights)
            print("epoch:", epoch)
            print("Loss in this step: ",
                  optimizer.loss_func.loss(X, Y, self.weights))

        print("======================================\n")
        print("Finished training with final loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        print("=====================================================\n")

    def predict(self, X):
        return np.dot(X, self.weights)

    def save(self, name):
        with open(name + '.rob', 'ab') as robfile:
            pickle.dump(self, robfile)
