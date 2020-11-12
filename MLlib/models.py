from MLlib.optimizers import GradientDescent
from MLlib.activations import sigmoid
from MLlib.utils.misc_utils import generate_weights
from MLlib.utils.decision_tree_utils import partition, find_best_split
from MLlib.utils.decision_tree_utils import Leaf, Decision_Node
from MLlib.utils .knn_utils import get_neighbours
from MLlib.utils.naive_bayes_utils import make_likelihood_table
from MLlib.utils.gaussian_naive_bayes_utils import get_mean_var, p_y_given_x
from MLlib.utils.k_means_clustering_utils import initi_centroid, cluster_allot
from MLlib.utils.k_means_clustering_utils import new_centroid, xy_calc
from collections import Counter
import numpy as np
import pickle
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

    fit(X,Y,optimizer=GradientDescent,epochs=25, \
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
        return sigmoid(prediction)

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
        Build the tree.

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
    pyx: P(y/X) is proportional to p(x1/y)*p(x2/y)...*p(y)
    using log and adding as multiplying for smaller
    numbers can make them very small
    As denominator P(X)=P(x1)*P(x2), is common we can ignore it.
    """

    def predict(self, x_label, y_class):

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

    # data is variable input given b user for which we predict the label.
    # Here we predict the gender from given list of height, weight, foot_size
    def predict(self, data,  x_label, y_class):

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

class Support_Vector_Machine:
    def __init__(self):
        pass

    def fit(self, data):
        self.data = data
        opt_dict = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [
                      self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,
                     ]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(
                                  -1*(self.max_feature_value*b_range_multiple),
                                  self.max_feature_value*b_range_multiple,
                                  step*b_multiple
                                  ):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("optimized a step")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])

            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification

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
        for i in range(1, epochs+1):
            interm = xy_calc(M, centroid_array)
            new_array = new_centroid(interm)
            centroid_array = new_array
        cluster_array = cluster_allot(M, centroid_array)
        for cluster in cluster_array:
            print("==============================\n")
            print(cluster)
            print("\n==============================\n")

