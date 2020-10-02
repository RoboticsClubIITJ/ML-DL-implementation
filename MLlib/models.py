from .optimizers import GradientDescent
from .utils.misc_utils import generate_weights
from .utils.decision_tree_utils import partition, find_best_split
from .utils.decision_tree_utils import Leaf, Decision_Node
import numpy as np
import pickle
from .activations import sigmoid


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


class DecisionTreeClassifier():

    root = None

    def fit(self, rows):
        """
        Builds the tree.

        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = partition(rows, question)

        # Recursively build the true branch.
        true_branch = self.fit(true_rows)

        # Recursively build the false branch.
        false_branch = self.fit(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        self.root = Decision_Node(question, true_branch, false_branch)

    def print_tree(self, spacing=""):
        """
        A tree printing function.
        """

        # Base case: we've reached a leaf
        if isinstance(self.root, Leaf):
            print(spacing + "Predict", self.root.predictions)
            return

        # Print the question at this node
        print(spacing + str(self.root.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(self.root.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(self.root.false_branch, spacing + "  ")

    def classify(self, row):
        """
        Classify a bit of data
        """

        # Base case: we've reached a leaf
        if isinstance(self.root, Leaf):
            return self.root.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if self.root.question.match(row):
            return self.classify(row, self.root.true_branch)
        else:
            return self.classify(row, self.root.false_branch)
