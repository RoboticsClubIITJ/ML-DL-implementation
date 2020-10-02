from .optimizers import GradientDescent
from .utils import generate_weights
import numpy as np
import pickle
from .activations import sigmoid


class LinearRegression():

    def fit(self, X, Y, optimizer=GradientDescent, epochs=25, zeros=False):

        self.weights = generate_weights(X.shape[1], 1, zeros=zeros)
        costs = []
        print("Starting training with loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        for epoch in range(1, epochs+1):
            print("======================================")
            self.weights = optimizer.iterate(X, Y, self.weights)
            print("epoch:", epoch)
            epoch_cost = optimizer.loss_func.loss(X, Y, self.weights)
            print("Loss in this step: ", epoch_cost)
            costs.append(epoch_cost)

        print("======================================\n")
        print("Finished training with final loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        print("=====================================================\n")
        return costs

    def predict(self, X):
        return np.dot(X, self.weights)

    def save(self, name):
        with open(name + '.rob', 'ab') as robfile:
            pickle.dump(self, robfile)


class LogisticRegression(LinearRegression):

    def predict(self, X):
        prediction = np.dot(X, self.weights).T
        return sigmoid(prediction)

    def classify(self, X):
        prediction = np.dot(X, self.weights).T
        prediction = sigmoid(prediction)
        actual_predictions = np.zeros((1, X.shape[0]))
        for i in range(prediction.shape[1]):
            if prediction[0][i] > 0.5:
                actual_predictions[0][i] = 1
        return actual_predictions
