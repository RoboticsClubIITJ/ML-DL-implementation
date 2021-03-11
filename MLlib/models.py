from MLlib.optimizers import GradientDescent
from MLlib.activations import Sigmoid
from MLlib.utils.misc_utils import generate_weights
from MLlib.utils.decision_tree_utils import partition, find_best_split
from MLlib.utils.decision_tree_utils import class_counts
from MLlib.utils .knn_utils import get_neighbours
from MLlib.utils.naive_bayes_utils import make_likelihood_table
from MLlib.utils.gaussian_naive_bayes_utils import get_mean_var, p_y_given_x
from MLlib.utils.k_means_clustering_utils import initi_centroid, cluster_allot
from MLlib.utils.k_means_clustering_utils import new_centroid, xy_calc
from MLlib.utils.pca_utils import PCA_utils, infer_dimension
from collections import Counter
import numpy as np
from numpy.random import random
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import math

DATE_FORMAT = '%d-%m-%Y_%H-%M-%S'


class LinearRegression():
    """
    Implement Linear Regression Model.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    fit(X,Y,optimizer=GradientDescent,epochs=25,
    zeros=False,save_best=False):
        Implement the Training of
        Linear Regression Model with
        suitable optimizer, inititalised
        random weights and Dataset's
        Input-Output, upto certain number
        of epochs.

    predict(X):
        Return the Predicted Value of
        Output associated with Input,
        using the weights, which were
        tuned by Training Linear Regression
        Model.

    save(name):
        Save the Trained Linear Regression
        Model in rob format , in Local
        disk.
    """

    def fit(
            self,
            X,
            Y,
            optimizer=GradientDescent,
            epochs=25,
            zeros=False,
            save_best=False
    ):
        """
        Train the Linear Regression Model
        by fitting its associated weights,
        according to Dataset's Inputs and
        their corresponding Output Values.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

        Y: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Output.

        optimizer: class
            Class of one of the Optimizers like
            AdamProp,SGD,MBGD,RMSprop,AdamDelta,
            Gradient Descent,etc.

        epochs: int
            Number of times, the loop to calculate loss
            and optimize weights, will going to take
            place.

        zeros: boolean
            Condition to initialize Weights as either
            zeroes or some random decimal values.

        save_best: boolean
            Condition to enable or disable the option
            of saving the suitable Weight values for the
            model after reaching the region nearby the
            minima of Loss-Function with respect to Weights.

        epoch_loss: float
            The degree of how much the predicted value
            is diverted from actual values, given by
            implementing one of choosen loss functions
            from loss_func.py .

        version: str
            Descriptive update of Model's Version at each
            step of Training Loop, along with Time description
            according to DATA_FORMAT.

        RETURNS
        =======

        None
        """
        self.weights = generate_weights(X.shape[1], 1, zeros=zeros)
        self.best_weights = {"weights": None, "loss": float('inf')}

        print("Starting training with loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        for epoch in range(1, epochs + 1):
            print("======================================")
            print("epoch:", epoch)
            self.weights = optimizer.iterate(X, Y, self.weights)
            epoch_loss = optimizer.loss_func.loss(X, Y, self.weights)
            if save_best and epoch_loss < self.best_weights["loss"]:
                print("updating best weights (loss: {})".format(epoch_loss))
                self.best_weights['weights'] = self.weights
                self.best_weights['loss'] = epoch_loss
                version = "model_best_" + datetime.now().strftime(DATE_FORMAT)
                print("Saving best model version: ", version)
                self.save(version)
            print("Loss in this step: ", epoch_loss)

        version = "model_final_" + datetime.now().strftime(DATE_FORMAT)
        print("Saving final model version: ", version)
        self.save(version)

        print("======================================\n")
        print("Finished training with final loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        print("=====================================================\n")

    def predict(self, X):
        """
        Predict the Output Value of
        Input, in accordance with
        Linear Regression Model.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            Predicted Values corresponding to
            each Input of Dataset.
        """
        return np.dot(X, self.weights)

    def save(self, name):
        """
        Save the Model in rob
        format for further usage.

        PARAMETERS
        ==========

        name: str
            Title of the Model's file
            to be saved in rob format.

        RETURNS
        =======

        None
        """
        with open(name + '.rob', 'wb') as robfile:
            pickle.dump(self, robfile)

    def plot(self, X, Y, optimizer=GradientDescent, epochs=25):
        """"
        Plot the graph of loss vs number of iterations

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
           1-D array of Dataset's input

        Y: ndarray(dtype=float, ndim=1)
           1-D array of Dataset's output

        optimizer: class
           Class of one of the Optimizers like
           AdamProp,SGD,MBGD,GradientDescent etc

        epochs: int
           Number of times, the loop to calculate loss
           and optimize weights, will going to take
           place.

        error: float
           The degree of how much the predicted value
           is diverted from actual values, given by implementing
           one of choosen loss functions from loss_func.py .

        RETURNS
        =========
        A 2-D graph with x-axis as Number of
        iterations and y-axis as loss.

        """
        l1 = []
        l2 = []
        self.weights = optimizer.loss_func.loss(X, Y, self.weights)
        for epoch in range(1, epochs + 1):
            l1.append(epoch)
            self.weights = optimizer.iterate(X, Y, self.weights)
            error = optimizer.loss_func.loss(X, Y, self.weights)
            l2.append(error)
        plt.plot(np.array(l1), np.array(l2))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


class PolynomialRegression():
    """
    Implement Polynomial Regression Model.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    fit(X,Y,optimizer=GradientDescent,epochs=60, \
    zeros=False,save_best=False):
        Implement the Training of
        Polynomial Regression Model with
        suitable optimizer, inititalised
        random weights and Dataset's
        Input-Output, upto certain number
        of epochs.

    predict(X):
        Return the Predicted Value of
        Output associated with Input,
        using the weights, which were
        tuned by Training Polynomial Regression
        Model.

    save(name):
        Save the Trained Polynomial Regression
        Model in rob format , in Local
        disk.
    """
    def __init__(self, degree):
        self.degree = degree
        self.weights = 0
        self.best_weights = {}

    def fit(
            self,
            X,
            Y,
            optimizer=GradientDescent,
            epochs=200,
            zeros=False,
            save_best=True
            ):
        """
        Train the Polynomial Regression Model
        by fitting its associated weights,
        according to Dataset's Inputs and
        their corresponding Output Values.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

            Update X with X**2, X**3, X**4 terms

        Y: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Output.

        optimizer: class
            Class of one of the Optimizers like
            AdamProp,SGD,MBGD,RMSprop,AdamDelta,
            Gradient Descent,etc.

        epochs: int
            Number of times, the loop to calculate loss
            and optimize weights, is going to take
            place.

        zeros: boolean
            Condition to initialize Weights as either
            zeroes or some random decimal values.

        save_best: boolean
            Condition to enable or disable the option
            of saving the suitable Weight values for the
            model after reaching the region nearby the
            minima of Loss-Function with respect to Weights.

        epoch_loss: float
            The degree of how much the predicted value
            is diverted from actual values, given by
            implementing one of choosen loss functions
            from loss_func.py .

        version: str
            Descriptive update of Model's Version at each
            step of Training Loop, along with Time description
            according to DATA_FORMAT.

        RETURNS
        =======

        None
        """
        M, N = X.shape

        P = X[:, 0:1]

        # Add polynomial terms to X
        # upto degree 'self.degree'.
        for i in range(2, self.degree+1):
            P = np.hstack((
                P,
                (np.power(X[:, 0:1], i)).reshape(M, 1)
            ))

        P = np.hstack((
            P,
            X[:, 1:2]
        ))

        X = P

        self.weights = generate_weights(X.shape[1], 1, zeros=zeros)
        self.best_weights = {"weights": self.weights, "loss":
                             optimizer.loss_func.loss(X, Y, self.weights)}
        print("Starting training with loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        for epoch in range(1, epochs + 1):
            print("======================================")
            print("epoch:", epoch)
            self.weights = optimizer.iterate(X, Y, self.weights)
            epoch_loss = optimizer.loss_func.loss(X, Y, self.weights)
            if save_best and epoch_loss < self.best_weights["loss"]:
                self.best_weights['weights'] = self.weights
                self.best_weights['loss'] = epoch_loss
                version = "model_best_" + datetime.now().strftime(DATE_FORMAT)
                print("Saving best model version: ", version)
                self.save(version)
            print("Loss in this step: ", epoch_loss)

        version = "model_final_" + datetime.now().strftime(DATE_FORMAT)
        print("Saving final model version: ", version)
        self.save(version)

        print("======================================\n")
        print("Finished training with final loss:", self.best_weights['loss'])
        print("=====================================================\n")

    def predict(self, X):
        """
        Predict the Output Value of
        Input, in accordance with
        Polynomial Regression Model.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

        RETURNS
        =======

        ndarray(dtype=float, ndim=1)
            Predicted Values corresponding to
            each Input of Dataset.
        """
        M, N = X.shape

        P = X[:, 0:1]

        for i in range(2, self.degree+1):
            P = np.hstack((
                P,
                (np.power(X[:, 0:1], i)).reshape(M, 1)
            ))

        P = np.hstack((
            P,
            X[:, 1:2]
        ))

        X = P

        return np.dot(X, self.best_weights['weights'])

    def save(self, name):
        """
        Save the Model in rob
        format for further usage.

        PARAMETERS
        ==========

        name: str
            Title of the Model's file
            to be saved in rob format.

        RETURNS
        =======

        None
        """
        with open(name + '.rob', 'wb') as robfile:
            pickle.dump(self, robfile)

    def plot(
            self,
            X,
            Y,
            Z,
            optimizer=GradientDescent,
            epochs=60,
            zeros=False,
            save_best=False
            ):

        """
        Plot the graph of Loss vs Epochs
        Plot the graph of line Of Polynomial Regression

        PARAMETERS
        ==========

        X: ndarray(dtype=float, ndim=1)
           1-D array of Dataset's input

        Y: ndarray(dtype=float, ndim=1)
           1-D array of Dataset's output

        Z: ndarray(dtype=float, ndim=1)
           1-D array of Predicted Values

        optimizer: class
            Class of one of the Optimizers like
            AdamProp,SGD,MBGD,RMSprop,AdamDelta,
            Gradient Descent,etc.

        epochs: int
            Number of times, the loop to calculate loss
            and optimize weights, is going to take
            place.

        zeros: boolean
            Condition to initialize Weights as either
            zeroes or some random decimal values.

        save_best: boolean
            Condition to enable or disable the option
            of saving the suitable Weight values for the
            model after reaching the region nearby the
            minima of Loss-Function with respect to Weights.

        RETURNS
        =======

        None
        """

        M, N = X.shape

        P = X[:, 0:1]

        for i in range(2, self.degree+1):
            P = np.hstack((
                P,
                (np.power(X[:, 0:1], i)).reshape(M, 1)
            ))

        P = np.hstack((
            P,
            X[:, 1:2]
            ))

        X = P
        m = []
        List = []
        self.weights = generate_weights(X.shape[1], 1, zeros=zeros)
        self.best_weights = {"weights": self.weights, "loss":
                             optimizer.loss_func.loss(X, Y, self.weights)}
        print("Starting training with loss:",
              optimizer.loss_func.loss(X, Y, self.weights))
        for epoch in range(1, epochs + 1):
            m.append(epoch)
            self.weights = optimizer.iterate(X, Y, self.weights)
            epoch_loss = optimizer.loss_func.loss(X, Y, self.weights)
            if save_best and epoch_loss < self.best_weights["loss"]:
                self.best_weights['weights'] = self.weights
                self.best_weights['loss'] = epoch_loss
            List.append(epoch_loss)
        x = np.array(m)
        y = np.array(List)
        plt.figure(figsize=(10, 5))
        plt.xlabel('EPOCHS', family='serif', fontsize=15)
        plt.ylabel('LOSS', family='serif', fontsize=15)
        plt.scatter(x, y, color='navy')
        plt.show()

        z = np.reshape(Z, (1, M))
        pred_value = z[0]
        true_value = Y[0]
        A = []
        for i in range(0, len(Y[0])):
            A.append(i)
        x_axis = np.array(A)
        plt.xlabel('Number of Datasets', family='serif', fontsize=15)
        plt.ylabel('Values', family='serif', fontsize=15)
        plt.scatter(x_axis, true_value, label="True Values")
        plt.plot(x_axis, pred_value, label="Predicted Values")
        plt.legend(loc="upper right")
        plt.show()


class LogisticRegression(LinearRegression):
    """
    Implements Logistic Regression Model.

    ATTRIBUTES
    ==========

    LinearRegression: Class
        Parent Class from where Output Prediction
        Value is expressed, after Linear Weighted
        Combination of Input is calculated .

    METHODS
    =======

    predict(X):
        Return the probabilistic value
        of an Input, belonging to either
        class 0 or class 1, by using final
        weights from Trained Logistic
        Regression Model.

    classify(X):
        Return the Class corresponding to
        each Input of Dataset, Predicted by
        Trained Logistic Regression Model,
        i.e in this scenario, either class 0
        or class 1.
    """

    def predict(self, X):
        """
        Predict the Probabilistic Value of
        Input, in accordance with
        Logistic Regression Model.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

        prediction: ndarray(dtype=float,ndim=1)
            1-D Array of Predicted Values
            corresponding to each Input of
            Dataset.

        RETURNS
        =======

        ndarray(dtype=float,ndim=1)
            1-D Array of Probabilistic Values
            of whether the particular Input
            belongs to class 0 or class 1.
        """
        prediction = np.dot(X, self.weights).T
        sigmoid = Sigmoid()
        return sigmoid.activation(prediction)

    def classify(self, X):
        """
        Classify the Input, according to
        Logistic Regression Model,i.e in this
        case, either class 0 or class 1.

        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
            1-D Array of Dataset's Input.

        prediction: ndarray(dtype=float,ndim=1)
            1-D Array of Predicted Values
            corresponding to their Inputs.

        actual_predictions: ndarray(dtype=int,ndim=1)
            1-D Array of Output, associated
            to each Input of Dataset,
            Predicted by Trained Logistic
            Regression Model.

        RETURNS
        =======

        ndarray
            1-D Array of Predicted classes
            (either 0 or 1) corresponding
            to their inputs.

        """
        prediction = np.dot(X, self.weights).T
        sigmoid = Sigmoid()
        prediction = sigmoid.activation(prediction)
        actual_predictions = np.zeros((1, X.shape[0]))
        for i in range(prediction.shape[1]):
            if prediction[0][i] > 0.5:
                actual_predictions[0][i] = 1

        return actual_predictions


class DecisionTreeClassifier():
    """
    A class to implement the Decision Tree Algorithm.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    print_tree(rows, head, spacing = "")
        To print the decision tree of the rows
        in an organised manner.

    classify(rows, head, prediction_val)
        To determine and return the predictions
        of the subsets of the dataset.
    """

    def print_tree(self, rows, head, spacing=""):
        """
        A tree printing function.

        PARAMETERS
        ==========

        rows: list
            A list of lists to store the dataset.

        head: list
            A list to store the headings of the
            columns of the dataset.

        spacing: String
            To store and update the spaces to
            print the tree in an organised manner.

        RETURNS
        =======

        None

        """

        # Try partitioning the dataset on each of the unique attribute,
        # calculate the gini impurity,
        # and return the question that produces the least gini impurity.
        gain, question = find_best_split(rows, head)

        # Base case: we've reached a leaf
        if gain == 0:
            print(spacing + "Predict", class_counts(rows, len(rows[0])-1))
            return

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = partition(rows, question)

        # Print the question at this node
        print(spacing + str(question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(true_rows, head, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(false_rows, head, spacing + "  ")

    def classify(self, rows, head, prediction_val):
        """
        A function to make predictions of
        the subsets of the dataset.

        PARAMETERS
        ==========

        rows: list
            A list of lists to store the subsets
            of the dataset.

        head: list
            A list to store the headings of the
            columns of the subset of the dataset.

        prediction_val: dictionary
            A dictionary to update and return the
            predictions of the subsets of the
            dataset.

        RETURNS
        =======

        prediction_val
            Dictionary to return the predictions
            corresponding to the subsets of the
            dataset.

        """

        N = len(rows[0])

        # Finding random indexes for columns
        # to collect random samples of the dataset.
        indexcol = []
        for j in range(0, 5):
            r = np.random.randint(0, N-2)
            if r not in indexcol:
                indexcol.append(r)

        row = []
        for j in rows:
            L = []
            for k in indexcol:
                L.append(j[k])
            row.append(L)

        # add last column to the random sample so created.
        for j in range(0, len(row)):
            row[j].append(rows[j][N-1])

        rows = row

        # Try partitioning the dataset on each of the unique attribute,
        # calculate the gini impurity,
        # and return the question that produces the least gini impurity.
        gain, question = find_best_split(rows, head)

        # Base case: we've reached a leaf
        if gain == 0:
            # Get the predictions of the current set of rows.
            p = class_counts(rows, len(rows[0])-1)
            for d in prediction_val:
                for j in p:
                    if d == j:
                        # update the predictions to be returned.
                        prediction_val[d] = prediction_val[d] + p[j]
            return prediction_val

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = partition(rows, question)

        # Recursively build the true branch.
        self.classify(true_rows, head, prediction_val)

        # Recursively build the false branch.
        self.classify(false_rows, head, prediction_val)

        # Return the dictionary of the predictions
        # at the end of the recursion.
        return prediction_val


class RandomForestClassifier(DecisionTreeClassifier):
    """
    A class to implement the Random Forest Classification Algorithm.

    ATTRIBUTES
    ==========

    DecisionTreeClassifier: Class
        Parent Class from where the predictions
        for the subsets of the dataset are made.

    METHODS
    =======

    predict(A, n_estimators=100):
        Print the value that appears the
        highest in the list of predictions
        of the subsets of the dataset.
    """

    def predict(self, A, head, n_estimators=100):
        """
        Determine the predictions of the
        subsets of the dataset through the
        DecisionTreeClassifier class and
        print the mode of the predicted values.

        PARAMETERS
        ==========

        A: ndarray(dtype=int,ndim=2)
            2-D Array of Dataset's Input

        n_estimators: int
            Number of Decision Trees to be
            iterated over for the classification.

        RETURNS
        =======

        None
        """

        prediction = {}
        print("Predictions of individual decision trees")
        # Iterate to collect predictions of
        # 100 Decision Trees after taking
        # random samples from the dataset.
        for i in range(n_estimators):
            M = len(A)

            # Finding random indexes for rows
            # to collect the bootstrapped samples
            # of the dataset.
            indexrow = np.random.randint(0, M-1, 6)
            rows = []
            for j in indexrow:
                rows.append(A[j])

            label = len(rows[0])-1

            # Get prediction values for the rows
            prediction_val = class_counts(rows, label)
            for d in prediction_val:
                prediction_val[d] = 0

            # Create object of class DecisionTreeClassifier
            RandomF = DecisionTreeClassifier()

            # Store the returned dictionary of the
            # predictions of the subsets of the dataset.
            di = RandomF.classify(rows, head, prediction_val)

            print(di)

            # find maximum predicted value for the subsets
            # of the dataset.
            maximum = 0
            for j in di:
                if di[j] > maximum:
                    maximum = di[j]
                    maxk = j

            # Update the dictionary prediction with the
            # maximum predicted value in the
            # predictions of the subsets of the dataset.
            if maxk not in prediction:
                prediction[maxk] = maximum
            else:
                prediction[maxk] = prediction[maxk]+maximum

        # find maximum predicted value, hence the
        # final prediction of the Random Forest Algorithm.
        maximum = 0
        for i in prediction:
            if prediction[i] > maximum:
                maximum = prediction[i]
                maxk = i

        # predicting the maximum occurence
        print("\n Predict = {", maxk, "}")


class KNN():
    """
    A single Class that can act as both KNN classifier or regressor
    based on arguements given to the prediction function.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    predict(train, test_row, num_neighbours=7, classify=True):
        K Nearest Neighbour Model, used as Classifier, to
        predict the class of test point , with respect to
        its n nearest neighbours.
    """

    def predict(self, train, test_row, num_neighbours=7, classify=True):
        """
        KNN Prediction Model, used for either Regression or
        Classification , in respect to Test Point and
        Dataset Type.

        PARAMETERS
        ==========

        train: ndarray
            Array Representation of Collection
            of Points, with their corresponding
            x1,x2 and y features.

        test_row: ndarray(dtype=int,ndim=1,axis=1)
            Array representation of test point,
            with its corresponding x1,x2 and y
            features.

        num_neighbours: int
            Number of nearest neighbours, close
            to the test point, with respect to
            x1,x2 and y features.

        classify: Boolean
            Type of Mode, K Nearest Neighbour
            Model wants to be applied, according
            to Dataset and Application Field.

        neighbours: list
            List of n nearest neighbours, close
            to the test point, with their
            associated Point Array and distance
            from the Test point.

        ouput: list
            List of Distances of n nearest
            neighbours, calculated with respect
            to the test point, using either
            Block or Euclidean Metric.

        key: int
            Count of number of terms inside
            ouput list.

        RETURNS
        =======

        prediction: float/int
            If used as a Classifier, gives
            Class number as prediction. Else,
            it will give the mean of Cluster
            made by test point and its n
            nearest neighbours.
        """

        neigbours = get_neighbours(
            train, test_row, num_neighbours, distance_metrics="block")
        ouput = [row[-1] for row in neigbours]
        if classify:
            prediction = max(set(ouput), key=ouput.count)
        else:
            prediction = sum(ouput) / len(ouput)
        return prediction


class Naive_Bayes():
    """
    A class which classifies and predicts based on simple
    Naive Bayes algorithm.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    predict(self, x_label, y_class):
        Naive Bayes Model to predict the
        class given the label.
    """

    def predict(self, x_label, y_class):
        """
        Naive Bayes Model to predict the
        class given the label.

        PARAMETERS
        ==========

        x_label: ndarray(dtype=int,ndim=1,axis=1)
                   Array of labels.

        y_class: ndarray(dtype=int,ndim=1,axis=1)
                  Array of classes.

        RETURNS
        =======

        Most probable output or prediction, as list
        of the label and class name.

        """

        pyx = []
        likelihood = make_likelihood_table(x_label, y_class)
        Y = np.unique(y_class)
        X = np.unique(x_label)
        for j in range(len(Y)):
            total = 0
            for i in range(len(X)):
                if(likelihood[i][j] == 0):
                    continue
                total += math.log(likelihood[i][j])
                y_sum = (y_class == Y[j]).sum()
                if y_sum:
                    total += math.log(y_sum / len(y_class))
                    pyx.append([total, X[i], Y[j]])

        prediction = max(pyx)

        return [prediction[1], prediction[2]]


class Gaussian_Naive_Bayes():
    """
    A class which classifies and predicts based on Gaussian
    Naive Bayes algorithm.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    predict(self, x_label, y_class):
        Gaussian Naive Bayes Model to predict the
        label for given class values.
    """

    # data is variable input given by user for which we predict the label.
    # Here we predict the gender from given list of height, weight, foot_size
    def predict(self, data,  x_label, y_class):
        """
        Gaussian Naive Bayes Model to predict the
        label given the class values.

        PARAMETERS
        ==========

        x_label: ndarray(dtype=int,ndim=1,axis=1)
                   Array of labels.

        y_class: ndarray(dtype=int,ndim=1,axis=1)
                  Array of classes.

        RETURNS
        =======

        Predicts the label, for given class values
        by user.

        """
        mean, var = get_mean_var(x_label, y_class)
        argmax = 0
        for (k1, v1), (k2, v2) in zip(mean.items(), var.items()):
            pre_prob = Counter(x_label)[k1] / len(x_label)
            pro = 1
            for i in range(len(v1)):
                pro *= p_y_given_x(data[i], v1[i], v2[i])
            pxy = pro * pre_prob
            if(pxy > argmax):
                prediction = k1
        return prediction


class BernoulliNB(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, x, y):

        separate = [[i for i, t in zip(x, y) if t == c] for c in np.unique(y)]
        count_for_sample = x.shape[0]
        self.class_log = [np.log(len(i)/count_for_sample) for i in separate]
        count = self.alpha + np.array([np.array(i).sum(axis=0) for i in
                                      separate])
        smoothing = 2 * self.alpha
        doc = np.array([smoothing + len(i) for i in separate])
        self.log_prob = count / doc[np.newaxis].T
        return self

    def predict_log(self, x):
        return [(np.log(self.log_prob) * i + np.log(1 - self.log_prob) *
                np.abs(i - 1)).sum(axis=1) + self.class_log for i in x]

    def predict(self, x):
        return np.argmax(self.predict_log(x), axis=1)


class MultinomialNB(object):

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, x, y):

        separate = [[i for i, t in zip(x, y) if t == c] for c in np.unique(y)]
        count_for_sample = x.shape[0]
        self.class_log = [np.log(len(i)/count_for_sample) for i in separate]
        count = self.alpha + np.array([np.array(i).sum(axis=0) for i in
                                       separate])
        self.log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log(self, x):
        return [(self.log_prob * i).sum(axis=1) + self.class_log for i in x]

    def predict(self, x):
        return np.argmax(self.predict_log(x), axis=1)


class KMeansClustering():
    """
    One of the models used for Unsupervised
    learning, by making finite number of clusters
    from Dataset points.

    ATTRIBUTES
    ==========

    None

    METHODS
    =======

    work(M, num_cluster, epochs):
        Give details about cluster arrangements
        from Dataset's Points, after suitable
        number of epoch steps.
    """

    def work(self, M, num_cluster, epochs):
        """
        Show the arrangement of clusters after
        certain  number of epochs, provided with
        number of clusters and Input Dataset
        Matrix.

        PARAMETERS
        ==========

        M: ndarray(dtype=int,ndim=2)
            Dataset Matrix with finite number
            of points, having their corresponding
            x and y coordinates.

        num_cluster: int
            Number of Clusters to be made from
            the provided Dataset's points.

        epochs: int
            Number of times, centroids' coordinates
            will change, to obtain suitable clusters
            with appropriate number of points.

        centroid_array: list
            List of randomly initialised centroids,
            out of Dataset points, which will be
            going to update with every epoch, in
            order to obtain suitable clusters.

        interm: ndarray(dtype=int,ndim=2)
            Intermediate Matrix, consisting of
            clusterwise sum of each coordinate,
            with number of points in each cluster.

        new_array: list
            Updated list of new centroids, due to
            changes in cluster points, with each
            epoch.

        cluster_array: list
            List of Resultant Clusters, made after
            updating centroids with each epoch.
            It consist of Centroid and its
            corresponding nearby points of each
            Cluster.

        cluster: list
            List of Current cluster to be shown
            on screen, with its corresponding
            centroid and nearby points.

        RETURNS
        =======

        None
        """
        centroid_array = initi_centroid(M, num_cluster)
        for i in range(1, epochs + 1):
            interm = xy_calc(M, centroid_array)
            new_array = new_centroid(interm)
            centroid_array = new_array
        cluster_array = cluster_allot(M, centroid_array)
        for cluster in cluster_array:
            print("==============================\n")
            print(cluster)
            print("\n==============================\n")


class Bayes_Optimization():
    # surrogate or approximation for the objective function
    def surrogate(self, model, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(X, return_std=True)

    def acquisition(self, X, Xsamples, model):
        yhat, _ = self.surrogate(model, X)
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs

    # optimize the acquisition function
    def opt_acquisition(self, X, y, model):
        # random search, generate random samples
        Xsamples = random(100)
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        scores = self.acquisition(X, Xsamples, model)
        # locate the index of the largest scores
        ix = np.argmax(scores)
        return Xsamples[ix, 0]

    # plot real observations vs surrogate function
    def plot(self, X, y, model):
        # scatter plot of inputs and real objective function
        plt.scatter(X, y)
        # line plot of surrogate function across domain
        Xsamples = np.asarray(np.arange(0, 1, 0.001))
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        ysamples, _ = self.surrogate(model, Xsamples)
        plt.plot(Xsamples, ysamples)
        # show the plot
        plt.show()


# ---------------------- Principle Component Analysis ------------------------


class PCA(PCA_utils):
    """
    Principal component analysis (PCA):
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    """
    def __init__(self, n_components=None, whiten=False, svd_solver='auto'):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.components = None
        self.mean = None
        self.explained_variances = None
        self.noise_variance = None
        self.fitted = False

    def fit(self, X, y=None):
        # fit the model with the data X
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X."""
        U, S, Vh = self._fit(X)
        U = U[:, :self.n_components]
        if self.whiten:
            U *= math.sqrt(X.shape[0] - 1)
        else:
            U *= S[:self.n_components]
        return U

    def _fit(self, X):
        '''Fitting function for the model'''
        # count the sparsity of the  ndarray
        count = np.count_nonzero(X)
        sparsity = 1.0 - (count/np.size(X))
        if sparsity > 0.5:
            raise TypeError('PCA does not support sparse input.')
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components
        fit_svd_solver = self.svd_solver
        if fit_svd_solver == 'auto':
            # Small problem or n_components == 'mle', call full PCA
            if max(X.shape) <= 500 or n_components == 'mle':
                fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                fit_svd_solver = 'randomized'
            # Case of n_components in (0,1)
            else:
                fit_svd_solver = 'full'
        # Call different fits for either full or truncated SVD
        if fit_svd_solver == 'full':
            return self.fit_full(X, n_components)
        else:
            raise ValueError("Unrecognized svd_solver="
                             "'{0}'".format(fit_svd_solver))

    def fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X."""
        n_samples, n_features = X.shape
        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only "
                                 "supported if n_samples >= n_features")
        # mean of the dataset
        self.mean = np.mean(X, axis=0)
        std = np.std(X)
        X = (X - self.mean) / std
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        # columns of U, rows of Vh
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        Vh *= signs[:, np.newaxis]
        components = Vh
        # explained variance by singular values
        explained_variances = (S**2)/(n_samples-1)
        explained_variance_ratio = (explained_variances /
                                    explained_variances.sum())
        singular_value = S.copy()
        if n_components == 'mle':
            n_components = infer_dimension(explained_variances, n_samples)
        elif 0 < n_components < 1.0:
            ratio_cumsum = np.cumsum(explained_variance_ratio, axis=None,
                                     dtype=np.float64)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1
        # Computing noise covariance using Probabilistic PCA model
        if n_components < min(n_features, n_samples):
            self.noise_variance = explained_variances[n_components:].mean()
        else:
            self.noise_variance = 1.0
        # storing the first n_component values
        self.components = components[:n_components]
        self.n_components = self.n_components
        self.explained_variances = explained_variances[:n_components]
        self.explained_variance_ratio = explained_variance_ratio[:n_components]
        self.singular_value = singular_value[:n_components]
        self.fitted = True
        return U, S, Vh

# ------------------------------Numerical Outliers Method----------------------


class Numerical_outliers():

    def get_percentile(c, percentile_rank):
        """
           get_percentile Function
           PARAMETER
           =========
           c:ndarray(dtype=float,ndim=1)
             input dataset
           percentile_rank: float type
           RETURNS
           =======
           Data corresponding to percentile rank
           """

        d = np.sort(c)
        index = int(((len(d) - 1) * percentile_rank) // 100)
        return d[index]

    def get_outliers(x):

        """ get_outliers Function
         PARAMETER
           =========
        x:ndarray(dtype=float,ndim=1)
            input dataset
         """

        Q1 = Numerical_outliers.get_percentile(x, 25)
        Q3 = Numerical_outliers.get_percentile(x, 75)
        iqr = Q3 - Q1
        lowerbound = Q1 - 1.5 * iqr
        upperbound = Q3 + 1.5 * iqr
        for i in range(len(x)):
            if x[i] > upperbound or x[i] < lowerbound:
                print("outlier=", x[i])
