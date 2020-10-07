import numpy as np

class Optimizer:
    """
        A class to perform Optimization of learning parameters.
        Available among ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"].
        Object of this class is made inside Compile method of stackking class.

        Example:-
        ----------
        self.optimizer = Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]
    """
    def __init__(self, layers, name=None, learning_rate = 0.01, mr=0.001):
        """
        layers:- It is the list of layers on model.
        name:- It is the type of Optimizer.
        learning_rate:- It is the learning rate given by user.
        mr:- It is a momentum rate. Often used on Gradient Descent.
        """
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        values = [self.sgd, self.iterative, self.momentum, self.rmsprop, self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]:values[i] for i in range(len(keys))}
        if name != None and name in keys:
            self.opt_dict[name](layers=layers, training=False)
            #pass
    def sgd(self, layers, learning_rate=0.01, beta=0.001, training=True):
        learning_rate = self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights += l.pdelta_weights*self.mr + l.delta_weights * learning_rate
                    l.biases += l.pdelta_biases*self.mr + l.delta_biases * learning_rate
                    l.pdelta_weights = l.delta_weights
                    l.pdelta_biases = l.delta_biases
                else:
                    l.pdelta_weights = 0
                    l.pdelta_biases = 0
                    #l.delta_weights = 0
                    #l.delta_biases = 0
    def iterative(self, layers, learning_rate=0.01, beta=0, training=True):
        for l in layers:
            if l.parameters !=0:
                l.weights -= learning_rate * l.delta_weights
                l.biases -= learning_rate * l.delta_biases
    def momentum(self, layers, learning_rate=0.1, beta1=0.9, weight_decay=0.0005, nesterov=True, training=True):
        learning_rate = self.learning_rate
        #beta1 = 1 - self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_momentum = beta1 * l.weights_momentum + learning_rate * l.delta_weights-weight_decay *learning_rate*l.weights
                    l.weights+=l.weights_momentum
                    #
                    l.biases_momentum = beta1 * l.biases_momentum + learning_rate * l.delta_biases-weight_decay *learning_rate*l.biases
                    l.biases+=l.biases_momentum
                else:
                    l.weights_momentum = 0
                    l.biases_momentum = 0

            
    def rmsprop(self, layers, learning_rate=0.001, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_rms = beta1*l.weights_rms + (1-beta1)*(l.delta_weights ** 2)
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_rms + epsilon))
                    l.biases_rms = beta1*l.biases_rms + (1-beta1)*(l.delta_biases ** 2)
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_rms + epsilon))
                else:
                    l.weights_rms = 0
                    l.biases_rms = 0
    def adagrad(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_adagrad += l.delta_weights ** 2
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_adagrad+epsilon))
                    l.biases_adagrad += l.delta_biases ** 2
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_adagrad+epsilon))
                else:
                    l.weights_adagrad = 0
                    l.biases_adagrad = 0
    def adam(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, training=True):
        #print(training)
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.t += 1
                    if l.t == 1:
                        l.pdelta_biases = 0
                        l.pdelta_weights = 0
                    l.weights_adam1 = beta1 * l.weights_adam1 + (1-beta1)*l.delta_weights
                    l.weights_adam2 = beta2 * l.weights_adam2 + (1-beta2)*(l.delta_weights**2)
                    mcap = l.weights_adam1/(1-beta1**l.t)
                    vcap = l.weights_adam2/(1-beta2**l.t)
                    l.delta_weights = mcap/(np.sqrt(vcap) + epsilon)
                    l.weights += l.pdelta_weights * self.mr + learning_rate * l.delta_weights
                    l.pdelta_weights = l.delta_weights * 0

                    l.biases_adam1 = beta1 * l.biases_adam1 + (1-beta1)*l.delta_biases
                    l.biases_adam2 = beta2 * l.biases_adam2 + (1-beta2)*(l.delta_biases**2)
                    mcap = l.biases_adam1/(1-beta1**l.t)
                    vcap = l.biases_adam2/(1-beta2**l.t)
                    l.delta_biases = mcap/(np.sqrt(vcap) +epsilon)
                    l.biases += l.pdelta_biases * self.mr + learning_rate * l.delta_biases
                    l.pdelta_biases = l.delta_biases * 0
                    
                else:
                    l.t = 0
                    l.weights_adam1 = 0
                    l.weights_adam2 = 0
                    l.biases_adam1 = 0
                    l.biases_adam2 = 0
                    
    def adamax(self, layers, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_m = beta1*l.weights_m + (1-beta1)*l.delta_weights
                    l.weights_v = np.maximum(beta2*l.weights_v, abs(l.delta_weights))
                    l.weights += (self.learning_rate/(1-beta1))*(l.weights_m/(l.weights_v+epsilon))
                    
                    l.biases_m = beta1*l.biases_m + (1-beta1)*l.delta_biases
                    l.biases_v = np.maximum(beta2*l.biases_v, abs(l.delta_biases))
                    l.biases += (self.learning_rate/(1-beta1))*(l.biases_m/(l.biases_v+epsilon))
                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0
                    
    def adadelta(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_v = beta1*l.weights_v + (1-beta1)*(l.delta_weights ** 2)
                    l.delta_weights = np.sqrt((l.weights_m+epsilon)/(l.weights_v+epsilon))*l.delta_weights
                    l.weights_m = beta1*l.weights_m + (1-beta1)*(l.delta_weights)
                    l.weights += l.delta_weights
                    
                    l.biases_v = beta1*l.biases_v + (1-beta1)*(l.delta_biases ** 2)
                    l.delta_biases = np.sqrt((l.biases_m+epsilon)/(l.biases_v+epsilon))*l.delta_biases
                    l.biases_m = beta1*l.biases_m+ (1-beta1)*(l.delta_biases)
                    l.biases += l.delta_biases
                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0