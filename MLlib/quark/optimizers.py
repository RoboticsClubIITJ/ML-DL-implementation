import numpy as np


class Optimizer:
    """
        A class to perform Optimization of learning parameters.
        Available among ["sgd", "iterative", "momentum", "rmsprop",
         "adagrad", "adam", "adamax", "adadelta"].
        Object of this class is made inside Compile method of stackking class.

        Example:-
        ----------
        self.optimizer = Optimizer(layers=self.layers,
         name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]
    """

    def __init__(self, layers, name=None, learning_rate=0.01, mr=0.001):
        """
        layers:- It is the list of layers on model.
        name:- It is the type of Optimizer.
        learning_rate:- It is the learning rate given by user.
        mr:- It is a momentum rate. Often used on Gradient Descent.
        """
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = [
            "sgd", "iterative", "momentum",
            "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        values = [
            self.sgd, self.iterative, self.momentum, self.rmsprop,
            self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]: values[i] for i in range(len(keys))}
        if name is not None and name in keys:
            self.opt_dict[name](layers=layers, training=False)

    def sgd(self, layers, learning_rate=0.01, beta=0.001, training=True):
        learning_rate = self.learning_rate
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights += lyr.pdelta_weights*self.mr + \
                        lyr.delta_weights * learning_rate
                    lyr.biases += lyr.pdelta_biases*self.mr + \
                        lyr.delta_biases * learning_rate
                    lyr.pdelta_weights = lyr.delta_weights
                    lyr.pdelta_biases = lyr.delta_biases
                else:
                    lyr.pdelta_weights = 0
                    lyr.pdelta_biases = 0

    def iterative(self, layers, learning_rate=0.01, beta=0, training=True):
        for lyr in layers:
            if lyr.parameters != 0:
                lyr.weights -= learning_rate * lyr.delta_weights
                lyr.biases -= learning_rate * lyr.delta_biases

    def momentum(
            self, layers, learning_rate=0.1,
            beta1=0.9, weight_decay=0.0005, nesterov=True, training=True):
        learning_rate = self.learning_rate
        # beta1 = 1 - self.learning_rate
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights_momentum = beta1 * lyr.weights_momentum + \
                        learning_rate * lyr.delta_weights-weight_decay * \
                        learning_rate*lyr.weights
                    lyr.weights += lyr.weights_momentum

                    lyr.biases_momentum = beta1 * lyr.biases_momentum + \
                        learning_rate * lyr.delta_biases-weight_decay * \
                        learning_rate * lyr.biases
                    lyr.biases += lyr.biases_momentum
                else:
                    lyr.weights_momentum = 0
                    lyr.biases_momentum = 0

    def rmsprop(
            self, layers, learning_rate=0.001,
            beta1=0.9, epsilon=1e-8, training=True):
        learning_rate = self.learning_rate
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights_rms = beta1*lyr.weights_rms + \
                        (1-beta1)*(lyr.delta_weights ** 2)
                    lyr.weights += learning_rate * \
                        (lyr.delta_weights/np.sqrt(lyr.weights_rms + epsilon))
                    lyr.biases_rms = beta1*lyr.biases_rms + \
                        (1-beta1)*(lyr.delta_biases ** 2)
                    lyr.biases += learning_rate * \
                        (lyr.delta_biases/np.sqrt(lyr.biases_rms + epsilon))
                else:
                    lyr.weights_rms = 0
                    lyr.biases_rms = 0

    def adagrad(
            self, layers, learning_rate=0.01,
            beta1=0.9, epsilon=1e-8, training=True):
        learning_rate = self.learning_rate
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights_adagrad += lyr.delta_weights ** 2
                    lyr.weights += learning_rate * \
                        (lyr.delta_weights/np.sqrt(
                            lyr.weights_adagrad+epsilon))
                    lyr.biases_adagrad += lyr.delta_biases ** 2
                    lyr.biases += learning_rate * \
                        (lyr.delta_biases/np.sqrt(lyr.biases_adagrad+epsilon))
                else:
                    lyr.weights_adagrad = 0
                    lyr.biases_adagrad = 0

    def adam(
            self, layers, learning_rate=0.001,
            beta1=0.9, beta2=0.999, epsilon=1e-8,
            decay=0, training=True):

        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.t += 1
                    if lyr.t == 1:
                        lyr.pdelta_biases = 0
                        lyr.pdelta_weights = 0
                    lyr.weights_adam1 = beta1 * lyr.weights_adam1 + \
                        (1-beta1)*lyr.delta_weights
                    lyr.weights_adam2 = beta2 * lyr.weights_adam2 + \
                        (1-beta2)*(lyr.delta_weights**2)
                    mcap = lyr.weights_adam1/(1-beta1**lyr.t)
                    vcap = lyr.weights_adam2/(1-beta2**lyr.t)
                    lyr.delta_weights = mcap/(np.sqrt(vcap) + epsilon)
                    lyr.weights += lyr.pdelta_weights * self.mr + \
                        learning_rate * lyr.delta_weights
                    lyr.pdelta_weights = lyr.delta_weights * 0

                    lyr.biases_adam1 = beta1 * lyr.biases_adam1 + \
                        (1-beta1)*lyr.delta_biases
                    lyr.biases_adam2 = beta2 * lyr.biases_adam2 + \
                        (1-beta2)*(lyr.delta_biases**2)
                    mcap = lyr.biases_adam1/(1-beta1**lyr.t)
                    vcap = lyr.biases_adam2/(1-beta2**lyr.t)
                    lyr.delta_biases = mcap/(np.sqrt(vcap) + epsilon)
                    lyr.biases += lyr.pdelta_biases * self.mr + \
                        learning_rate * lyr.delta_biases
                    lyr.pdelta_biases = lyr.delta_biases * 0

                else:
                    lyr.t = 0
                    lyr.weights_adam1 = 0
                    lyr.weights_adam2 = 0
                    lyr.biases_adam1 = 0
                    lyr.biases_adam2 = 0

    def adamax(
            self, layers, learning_rate=0.002,
            beta1=0.9, beta2=0.999, epsilon=1e-8,
            training=True):
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights_m = beta1*lyr.weights_m + \
                        (1-beta1)*lyr.delta_weights
                    lyr.weights_v = np.maximum(
                        beta2*lyr.weights_v, abs(lyr.delta_weights))
                    lyr.weights += (self.learning_rate/(1-beta1))*(
                        lyr.weights_m/(lyr.weights_v+epsilon))

                    lyr.biases_m = beta1*lyr.biases_m + \
                        (1-beta1)*lyr.delta_biases
                    lyr.biases_v = np.maximum(
                        beta2*lyr.biases_v, abs(lyr.delta_biases))
                    lyr.biases += (self.learning_rate/(1-beta1))*(
                        lyr.biases_m/(lyr.biases_v+epsilon))

                else:
                    lyr.weights_m = 0
                    lyr.biases_m = 0
                    lyr.weights_v = 0
                    lyr.biases_v = 0

    def adadelta(
            self,
            layers, learning_rate=0.01, beta1=0.9,
            epsilon=1e-8, training=True):
        for lyr in layers:
            if lyr.parameters != 0:
                if training:
                    lyr.weights_v = beta1*lyr.weights_v + \
                        (1-beta1)*(lyr.delta_weights ** 2)
                    lyr.delta_weights = np.sqrt((lyr.weights_m+epsilon)/(
                        lyr.weights_v+epsilon))*lyr.delta_weights
                    lyr.weights_m = beta1*lyr.weights_m + \
                        (1-beta1)*(lyr.delta_weights)
                    lyr.weights += lyr.delta_weights

                    lyr.biases_v = beta1*lyr.biases_v + \
                        (1-beta1)*(lyr.delta_biases ** 2)
                    lyr.delta_biases = np.sqrt((lyr.biases_m+epsilon)/(
                        lyr.biases_v+epsilon))*lyr.delta_biases
                    lyr.biases_m = beta1*lyr.biases_m + \
                        (1-beta1)*(lyr.delta_biases)
                    lyr.biases += lyr.delta_biases

                else:
                    lyr.weights_m = 0
                    lyr.biases_m = 0
                    lyr.weights_v = 0
                    lyr.biases_v = 0
